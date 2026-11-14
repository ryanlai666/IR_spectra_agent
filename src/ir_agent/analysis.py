"""Deterministic analysis: match computed modes to reference fundamentals, find key bands,
and render a Markdown report (the LLM analyst only adds a narrative paragraph on top)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from ir_agent.validation.reference import (
    ADSORBATE_REFERENCES,
    expanded_fundamentals,
    gas_phase_fundamentals,
)
from ir_agent.vibrations.ir import IRResult


@dataclass
class ModeMatch:
    mode: str
    symmetry: str
    exp_cm: float
    calc_cm: float
    calc_intensity: float | None
    exp_intensity: float | None
    ir_active: bool

    @property
    def error_cm(self) -> float:
        return self.calc_cm - self.exp_cm

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "error_cm": self.error_cm}


def match_gas_phase(name: str, result: IRResult, min_cm: float = 50.0) -> list[ModeMatch] | None:
    """Assign computed modes to experimental fundamentals by frequency order.

    Degenerate experimental modes are expanded, matched in ascending order, then averaged.
    Returns None if there is no reference or the mode count does not match.
    """
    refs = expanded_fundamentals(name)
    if not refs:
        return None
    mask = result.frequencies_cm > min_cm
    freqs = result.frequencies_cm[mask]
    ints = None if result.intensities_km_mol is None else result.intensities_km_mol[mask]
    if len(freqs) < len(refs):
        return None
    order = np.argsort(freqs)[-len(refs):]  # highest 3N-6 real modes
    freqs, ints = freqs[order], (None if ints is None else ints[order])
    ref_order = np.argsort([r.wavenumber_cm for r in refs])
    grouped: dict[int, list[int]] = {}
    for k, ri in enumerate(ref_order):
        grouped.setdefault(id(refs[ri]), []).append(k)
    out: list[ModeMatch] = []
    for f in gas_phase_fundamentals(name) or []:
        ks = grouped[id(f)]
        calc = float(np.mean(freqs[ks]))
        inten = None if ints is None else float(np.sum(ints[ks]))
        out.append(ModeMatch(f.mode, f.symmetry, f.wavenumber_cm, calc, inten,
                             f.ir_intensity_km_mol, f.ir_active))
    return out


def optimal_scale_factor(matches: list[ModeMatch]) -> float:
    """Least-squares scale lambda minimizing sum (lambda*calc - exp)^2."""
    c = np.array([m.calc_cm for m in matches])
    e = np.array([m.exp_cm for m in matches])
    return float((c @ e) / (c @ c)) if c.size else 1.0


def error_stats(matches: list[ModeMatch], scale: float = 1.0) -> dict[str, float]:
    err = np.array([scale * m.calc_cm - m.exp_cm for m in matches])
    return {
        "mae_cm": float(np.abs(err).mean()),
        "max_abs_cm": float(np.abs(err).max()),
        "mse_cm": float(err.mean()),
        "mape_pct": float(np.mean(np.abs(err) / np.array([m.exp_cm for m in matches])) * 100),
    }


# --------------------------------------------------------------------------- band selection


def _cart_modes(result: IRResult, atoms: Atoms) -> np.ndarray:
    m = atoms.get_masses()[result.indices]
    inv = np.repeat(1.0 / np.sqrt(m), 3)
    disp = (result.modes_mw * inv[:, None]).T  # (n_modes, 3n)
    return disp.reshape(disp.shape[0], -1, 3)


def find_band(
    result: IRResult, atoms: Atoms, band: str, min_cm: float = 50.0
) -> tuple[int, float] | None:
    """Return (mode index, wavenumber) of a named band: 'C-O stretch' or 'umbrella'."""
    if result.modes_mw is None:
        return None
    disp = _cart_modes(result, atoms)
    syms = [atoms[i].symbol for i in result.indices]
    pos = atoms.get_positions()[result.indices]
    freqs = result.frequencies_cm
    scores = np.zeros(len(freqs))
    if band.startswith("C-O"):
        if "C" not in syms or "O" not in syms:
            return None
        ic, io = syms.index("C"), syms.index("O")
        u = pos[ic] - pos[io]
        u /= np.linalg.norm(u)
        for k in range(len(freqs)):
            d = disp[k]
            scores[k] = abs((d[ic] - d[io]) @ u) / (np.linalg.norm(d) + 1e-12)
    elif "umbrella" in band:
        if "N" not in syms:
            return None
        hs = [i for i, s in enumerate(syms) if s == "H"]
        n_i = syms.index("N")
        axis = pos[hs].mean(0) - pos[n_i]
        axis /= np.linalg.norm(axis)
        for k in range(len(freqs)):
            d = disp[k]
            rel = sum((d[h] - d[n_i]) @ axis for h in hs)
            scores[k] = abs(rel) / (np.linalg.norm(d) + 1e-12)
        scores[(freqs < 600) | (freqs > 1600)] = 0.0
    else:
        raise ValueError(f"unknown band '{band}'")
    scores[freqs < min_cm] = 0.0
    if scores.max() <= 0:
        return None
    k = int(np.argmax(scores))
    return k, float(freqs[k])


# --------------------------------------------------------------------------- report


def render_report(
    *,
    plan_desc: str,
    calculator: str,
    result: IRResult,
    atoms: Atoms,
    adsorbate: str,
    metal: str | None,
    min_cm: float = 50.0,
    scale_factor: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Markdown report + machine-readable summary."""
    lines = [f"# IR report: {plan_desc}", ""]
    lines.append(f"- Force/Hessian method: `{result.force_method}`")
    lines.append(f"- Dipole method: `{result.dipole_method or 'none (frequencies only)'}`")
    lines.append(f"- Route: `{result.route}`; displaced geometries: {result.n_displacements}")
    if result.surface_selection_rule:
        lines.append("- Metal-surface selection rule: on (only the surface-normal dipole derivative)")
    if scale_factor != 1.0:
        lines.append(f"- Frequencies scaled by {scale_factor}")
    lines.append(f"- Timings (s): {', '.join(f'{k}={v:.2f}' for k, v in result.timings_s.items())}")
    for note in result.notes:
        lines.append(f"- Note: {note}")
    lines += ["", "## Modes", "", "| # | wavenumber (cm^-1) | intensity (km/mol) |", "|---|---:|---:|"]
    f = result.frequencies_cm * scale_factor
    for k in np.argsort(f):
        if result.frequencies_cm[k] <= min_cm:
            continue
        inten = "n/a" if result.intensities_km_mol is None else f"{result.intensities_km_mol[k]:.2f}"
        lines.append(f"| {k} | {f[k]:.1f} | {inten} |")
    n_imag = int((result.frequencies_cm < -min_cm).sum())
    if n_imag:
        lines.append(f"\n{n_imag} imaginary mode(s) below -{min_cm} cm^-1: geometry not a minimum.")

    summary: dict[str, Any] = {"n_imaginary": n_imag, "calculator": calculator}
    if metal is None:
        matches = match_gas_phase(adsorbate, result, min_cm)
        if matches:
            lines += ["", "## Comparison with experiment (NIST CCCBDB fundamentals)", "",
                      "| mode | sym | exp | calc | error | I_calc (km/mol) | I_exp |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for m in matches:
                ic = "n/a" if m.calc_intensity is None else f"{m.calc_intensity:.1f}"
                ie = "-" if m.exp_intensity is None else f"{m.exp_intensity:.1f}"
                lines.append(f"| {m.mode} | {m.symmetry} | {m.exp_cm:.0f} | {m.calc_cm*scale_factor:.0f} | "
                             f"{m.calc_cm*scale_factor - m.exp_cm:+.0f} | {ic} | {ie} |")
            stats = error_stats(matches, scale_factor)
            lam = optimal_scale_factor(matches)
            lines.append(f"\nMAE {stats['mae_cm']:.0f} cm^-1 (harmonic vs anharmonic fundamentals); "
                         f"best-fit scale factor for this molecule {lam:.3f}.")
            summary.update(stats=stats, matches=[m.as_dict() for m in matches], best_scale=lam)
    else:
        ref = ADSORBATE_REFERENCES.get((adsorbate, metal))
        if ref is not None:
            found = find_band(result, atoms, ref.band, min_cm)
            lines += ["", f"## Literature trend: {ref.system}", ""]
            lines.append(f"- Band: {ref.band}; gas-phase fundamental {ref.gas_cm:.0f} cm^-1; "
                         f"adsorbed {'n/a' if ref.adsorbed_cm is None else f'{ref.adsorbed_cm:.0f}'} cm^-1")
            lines.append(f"- Expected trend: {ref.trend}")
            lines.append(f"- Source: {ref.source}{'' if ref.verified else ' (not fully verified)'}")
            if ref.shift_window:
                lo, hi = ref.shift_window
                lines.append(f"- A computed harmonic shift (vs the same method's gas value) between {lo:+.0f} and "
                             f"{hi:+.0f} cm^-1 counts as consistent. {ref.note}")
            if found:
                k, nu = found
                lines.append(f"- Computed {ref.band}: {nu*scale_factor:.0f} cm^-1 (mode {k})")
                summary["band_cm"] = nu * scale_factor
                summary["band"] = ref.band
    if extra:
        summary.update(extra)
    return "\n".join(lines) + "\n", summary
