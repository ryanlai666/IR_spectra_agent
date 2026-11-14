"""Matplotlib figures: IR spectrum and geometry progression."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ase import Atoms  # noqa: E402

from ir_agent.runtime import get_logger  # noqa: E402
from ir_agent.spectrum import Spectrum  # noqa: E402

log = get_logger("plotting")


def plot_spectrum(
    spectrum: Spectrum,
    path: Path,
    title: str,
    label: str = "",
    reference_cm: Sequence[float] | None = None,
    reference_label: str = "experiment",
) -> Path:
    """Broadened spectrum + sticks (+ optional dashed reference lines)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spectrum.x_cm, spectrum.y, color="#3b4cc0", lw=1.6, label=label or "calculated")
    if spectrum.stick_intensities.size:
        ymax = max(spectrum.y.max(), 1e-12)
        smax = max(spectrum.stick_intensities.max(), 1e-12)
        ax.vlines(spectrum.sticks_cm, 0, spectrum.stick_intensities / smax * ymax * 0.9,
                  color="#3b4cc0", alpha=0.35, lw=1.0)
    if reference_cm:
        for i, r in enumerate(reference_cm):
            ax.axvline(r, color="#b40426", ls="--", lw=0.9, label=reference_label if i == 0 else None)
    ax.set_xlim(spectrum.x_cm.max(), spectrum.x_cm.min())  # IR convention: high -> low
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    y_lab = "Absorbance (km mol$^{-1}$ / cm$^{-1}$)" if spectrum.y_units.startswith("km") else "Mode density (arb. units, no dipoles)"
    ax.set_ylabel(y_lab)
    ax.set_title(title)
    ax.text(0.99, 0.97, f"{spectrum.lineshape}, FWHM {spectrum.fwhm_cm:g} cm$^{{-1}}$"
            + (f", scale {spectrum.scale_factor:g}" if spectrum.scale_factor != 1 else ""),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_structures(stages: dict[str, Atoms], path: Path, title: str = "Geometry progression") -> Path:
    """Side (-90x) and top views of each stage (generalizes the v0.1 2x4 comparison plot)."""
    from ase.visualize.plot import plot_atoms

    names = list(stages)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.5), squeeze=False)
    for j, name in enumerate(names):
        atoms = stages[name]
        for i, rot in enumerate(("-90x", "0x")):
            ax = axes[i, j]
            try:
                plot_atoms(atoms, ax, rotation=rot, radii=0.8)
            except Exception as exc:  # pragma: no cover - plotting never fails the run
                log.warning("plot_atoms failed for %s: %s", name, exc)
            ax.set_title(f"{name.replace('_', ' ')} ({'side' if i == 0 else 'top'})", fontsize=10)
            ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
