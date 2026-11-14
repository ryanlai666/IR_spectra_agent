"""Evaluators that map a list of displaced geometries to forces or dipoles.

A *force evaluator* returns ``(n_images, n_atoms, 3)`` forces in eV/A; a *dipole evaluator*
returns ``(n_images, 3)`` dipoles in e*A. Implementations may batch (UMA on GPU, dxtb) or
loop over an ordinary ASE calculator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from ir_agent.runtime import get_logger

log = get_logger("evaluators")


@runtime_checkable
class ForceEvaluator(Protocol):
    name: str

    def __call__(self, images: Sequence[Atoms]) -> np.ndarray: ...


@runtime_checkable
class DipoleEvaluator(Protocol):
    name: str

    def __call__(self, images: Sequence[Atoms]) -> np.ndarray: ...


class ASECalculatorForces:
    """Loop over images with one ASE calculator (works for any calculator)."""

    def __init__(self, calc: Calculator | Callable[[], Calculator], name: str | None = None):
        self._calc = calc
        self.name = name or getattr(calc, "name", type(calc).__name__)

    def _get_calc(self) -> Calculator:
        return self._calc() if callable(self._calc) and not isinstance(self._calc, Calculator) else self._calc

    def __call__(self, images: Sequence[Atoms]) -> np.ndarray:
        calc = self._get_calc()
        out = []
        for k, img in enumerate(images):
            atoms = img.copy()
            atoms.calc = calc
            out.append(atoms.get_forces(apply_constraint=False))
            if (k + 1) % 20 == 0:
                log.debug("forces %d/%d", k + 1, len(images))
        return np.asarray(out)


class ASECalculatorDipoles:
    """Dipoles from an ASE calculator implementing ``get_dipole_moment``."""

    def __init__(
        self,
        calc: Calculator | Callable[[], Calculator],
        name: str | None = None,
        region: Sequence[int] | None = None,
        cluster_builder: Callable[[Atoms], Atoms] | None = None,
    ):
        """
        Args:
            calc: dipole-capable calculator (or a zero-argument factory).
            region: if given, evaluate the dipole of this atom subset only (as a non-periodic
                fragment), e.g. the adsorbate on a slab whose force method has no dipoles.
            cluster_builder: alternative to ``region``; maps a full image to the cluster whose
                dipole is evaluated (must keep atom ordering consistent between images).
        """
        self._calc = calc
        self.name = name or getattr(calc, "name", type(calc).__name__)
        self.region = list(region) if region is not None else None
        self.cluster_builder = cluster_builder

    def _get_calc(self) -> Calculator:
        return self._calc() if callable(self._calc) and not isinstance(self._calc, Calculator) else self._calc

    def _prepare(self, img: Atoms) -> Atoms:
        if self.cluster_builder is not None:
            return self.cluster_builder(img)
        if self.region is not None:
            return extract_fragment(img, self.region)
        atoms = img.copy()
        atoms.calc = None
        return atoms

    def __call__(self, images: Sequence[Atoms]) -> np.ndarray:
        calc = self._get_calc()
        out = []
        for img in images:
            atoms = self._prepare(img)
            atoms.calc = calc
            out.append(np.asarray(atoms.get_dipole_moment(), dtype=float))
        return np.asarray(out)


class FixedChargeDipoles:
    """mu = sum_i q_i r_i with fixed point charges (toy model / unit tests)."""

    name = "fixed-charges"

    def __init__(self, charges: Sequence[float]):
        self.charges = np.asarray(charges, dtype=float)

    def __call__(self, images: Sequence[Atoms]) -> np.ndarray:
        return np.array([self.charges @ img.get_positions() for img in images])


def extract_fragment(atoms: Atoms, indices: Sequence[int]) -> Atoms:
    """Copy ``atoms[indices]`` as an isolated, non-periodic fragment (positions unchanged)."""
    frag = atoms[list(indices)].copy()
    frag.set_constraint()
    frag.calc = None
    frag.pbc = False
    frag.info = {k: v for k, v in atoms.info.items() if k in ("charge", "spin")}
    return frag


class BatchedForces:
    """Base for MLIP evaluators that push many displaced geometries through one forward pass.

    Subclasses implement :meth:`predict_chunk` (images -> stacked forces ``(sum n_atoms, 3)``).
    All images of a finite-difference run share composition, so a pass holds
    ``effective_batch_size`` images, capped by ``max_atoms_per_batch`` to bound peak VRAM.
    """

    name = "batched"

    def __init__(self, batch_size: int = 16, max_atoms_per_batch: int | None = None):
        self.batch_size = max(1, int(batch_size))
        self.max_atoms_per_batch = max_atoms_per_batch
        self.n_forward_passes = 0

    def effective_batch_size(self, n_atoms: int) -> int:
        if not self.max_atoms_per_batch:
            return self.batch_size
        return max(1, min(self.batch_size, self.max_atoms_per_batch // max(1, n_atoms)))

    def prepare(self, img: Atoms) -> Atoms:
        """Hook to set per-image metadata (charge/spin) before prediction."""
        return img

    def predict_chunk(self, images: Sequence[Atoms]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, images: Sequence[Atoms]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0, 3))
        n = len(images[0])
        bs = self.effective_batch_size(n)
        out: list[np.ndarray] = []
        for start in range(0, len(images), bs):
            chunk = [self.prepare(img.copy()) for img in images[start : start + bs]]
            forces = np.asarray(self.predict_chunk(chunk), dtype=float)
            if forces.shape != (n * len(chunk), 3):
                raise RuntimeError(f"Unexpected force array {forces.shape} for {len(chunk)} images of {n} atoms")
            out.extend(forces.reshape(len(chunk), n, 3))
            self.n_forward_passes += 1
        return np.asarray(out)
