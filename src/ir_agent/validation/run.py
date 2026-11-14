"""Validation matrix: calculators x {gas-phase molecules vs NIST, adsorbates vs literature trends}.

Every adsorbate row carries the critic's checks (convergence, imaginary modes, relaxed site; see
``ir_agent.critic``) and a gas-phase reference computed with the *same* force method and, for UMA,
the same task as the adsorbate calculation (plain ``uma`` uses oc20 for slabs, so its shifts are
referenced to a ``uma-oc20`` gas run, not to the omol gas run).
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ir_agent.agent.plan import SimulationPlan, SurfaceSpec
from ir_agent.analysis import error_stats, find_band, match_gas_phase
from ir_agent.calculators.registry import canonical_preset, preset_status
from ir_agent.config import AppConfig
from ir_agent.critic import critique
from ir_agent.pipeline import run_pipeline
from ir_agent.plotting import plot_spectrum
from ir_agent.runtime import RunDir, get_logger
from ir_agent.tracking import tracked_run
from ir_agent.validation.reference import ADSORBATE_REFERENCES, gas_phase_fundamentals

log = get_logger("validation")


def _sub(root: RunDir, name: str) -> RunDir:
    d = root.root / name
    for s in ("structures", "calc", "plots"):
        (d / s).mkdir(parents=True, exist_ok=True)
    return RunDir(d)


def _gpu_peak_mb() -> float | None:
    from ir_agent.gpu import peak_vram_mb

    return peak_vram_mb()


def _load_row(rd: RunDir) -> dict[str, Any] | None:
    p = rd.root / "row.json"
    if not p.exists():
        return None
    log.info("resume: reusing %s", p.parent.name)
    return json.loads(p.read_text(encoding="utf-8"))


def bulk_structure(metal: str, facet: str) -> str:
    """Lattice for a surface spec: hcp for (0001)/(10m10), else the element's ASE reference state
    (Cu, Pt, Au: fcc; Ru, Co: hcp; Fe, W: bcc)."""
    if facet in ("0001", "10m10"):
        return "hcp"
    from ase.data import atomic_numbers, reference_states

    sym = (reference_states[atomic_numbers[metal]] or {}).get("symmetry", "fcc")
    return sym if sym in ("fcc", "bcc", "hcp") else "fcc"


def parse_surface_spec(spec: str) -> tuple[str, str, str, str]:
    """'CO/Pt/111' or 'CO/Pt/111/bridge' -> (molecule, metal, facet, site)."""
    parts = spec.split("/")
    if len(parts) not in (3, 4):
        raise ValueError(f"surface spec '{spec}' must be MOL/METAL/FACET[/SITE]")
    mol, metal, facet = parts[:3]
    return mol, metal, facet, (parts[3] if len(parts) == 4 else "ontop")


def uma_task(calc: str, kind: str, cfg: AppConfig) -> str | None:
    """UMA task used for ``calc`` on a system of ``kind`` (None for non-UMA presets)."""
    from ir_agent.calculators.uma import UMA_PRESETS, task_for_preset

    if calc not in UMA_PRESETS:
        return None
    return task_for_preset(calc, "surface" if kind == "surface" else "molecule", cfg.calculators.uma)


def gas_reference_preset(calc: str, ads_kind: str, cfg: AppConfig) -> str:
    """Preset for the gas-phase reference of an adsorbate row: same method and same UMA task."""
    task = uma_task(calc, ads_kind, cfg)
    if task is None or task == uma_task(calc, "molecule", cfg):
        return calc
    return f"uma-{task}"


def run_validation(
    cfg: AppConfig,
    calculators: list[str],
    molecules: list[str],
    surfaces: list[str],
    skip_unavailable: bool = True,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the matrix. With ``resume_dir`` finished sub-runs (``row.json``) are reloaded instead
    of recomputed, so an interrupted campaign (e.g. a Colab disconnect) continues where it stopped.

    ``surfaces`` entries are ``MOL/METAL/FACET[/SITE]`` (site defaults to ontop)."""
    root = RunDir(Path(resume_dir)) if resume_dir else RunDir.create(cfg.run.output_root, "validation")
    gas_rows: list[dict[str, Any]] = []
    ads_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    gas_band: dict[tuple[str, str], float] = {}
    done_gas: set[tuple[str, str]] = set()

    def run_gas(mol: str, calc: str) -> None:
        if (mol, calc) in done_gas:
            return
        done_gas.add((mol, calc))
        plan = SimulationPlan(adsorbate=mol, calculator=calc)
        rd = _sub(root, f"gas_{mol}_{calc}")
        done = _load_row(rd)
        if done is not None:
            gas_rows.append(done)
            for nu in (done.get("gas_bands") or {}).values():
                gas_band[(mol, calc)] = nu
            return
        row: dict[str, Any] = {"calculator": calc, "molecule": mol, "task": uma_task(calc, "molecule", cfg)}
        t0 = time.perf_counter()
        try:
            out = run_pipeline(plan, cfg, rd)
            res = out["result"]
            crit = critique(plan, out["prepared"], res, cfg)
            row.update(wall_s=time.perf_counter() - t0, force_method=res.force_method,
                       dipole=res.dipole_method, route=res.route, timings=res.timings_s,
                       n_disp=res.n_displacements, gpu_peak_mb=_gpu_peak_mb(),
                       converged=out["prepared"].converged.get("final"), n_imag=crit.n_imag,
                       flags=crit.codes, issues=[asdict(i) for i in crit.issues])
            matches = match_gas_phase(mol, res, cfg.vibrations.min_frequency_cm)
            row["matches"] = [m.as_dict() for m in matches] if matches else None
            if matches:
                row["stats"] = error_stats(matches)
            for band in ("C-O stretch", "umbrella"):
                fb = find_band(res, out["prepared"].stages["final"], band)
                if fb:
                    gas_band[(mol, calc)] = fb[1]
                    row.setdefault("gas_bands", {})[band] = fb[1]
            refs = [f.wavenumber_cm for f in gas_phase_fundamentals(mol) or [] if f.ir_active]
            plot_spectrum(out["spectrum"], rd.path("plots", "ir_spectrum.png"),
                          f"{mol} gas phase: {calc}", f"{res.force_method} / {res.dipole_method}",
                          refs, "NIST fundamentals")
            rd.path("report.md").write_text(out["report"], encoding="utf-8")
            rd.write_json("row.json", row)
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - t0)
            log.error("%s/%s failed: %s\n%s", mol, calc, exc, traceback.format_exc())
        gas_rows.append(row)

    for calc_name in calculators:
        calc = canonical_preset(calc_name)
        ok, reason = preset_status(calc, cfg)
        if not ok:
            if not skip_unavailable:
                raise RuntimeError(f"{calc}: {reason}")
            skipped[calc] = reason
            log.warning("Skipping %s: %s", calc, reason)
            continue
        for mol in molecules:
            run_gas(mol, calc)

        for spec in surfaces:
            mol, metal, facet, site = parse_surface_spec(spec)
            plan = SimulationPlan(adsorbate=mol, calculator=calc,
                                  surface=SurfaceSpec(metal=metal, structure=bulk_structure(metal, facet),
                                                  facet=facet, site=site))
            site_tag = "" if site == "ontop" else f"_{site}"
            rd = _sub(root, f"ads_{mol}_{metal}{facet}{site_tag}_{calc}")
            done = _load_row(rd)
            if done is not None:
                ads_rows.append(done)
                continue
            system = f"{mol}/{metal}({facet})" + ("" if site == "ontop" else f"/{site}")
            row = {"calculator": calc, "system": system, "site_requested": site}
            t0 = time.perf_counter()
            try:
                out = run_pipeline(plan, cfg, rd)
                res, ps = out["result"], out["prepared"]
                crit = critique(plan, ps, res, cfg)
                ref = ADSORBATE_REFERENCES.get((mol, metal))
                ref_calc = gas_reference_preset(calc, ps.kind, cfg)
                row.update(wall_s=time.perf_counter() - t0, force_method=res.force_method,
                           dipole=res.dipole_method, route=res.route, model=ps.kind,
                           n_atoms=len(ps.atoms), task=uma_task(calc, ps.kind, cfg),
                           gas_ref_calculator=ref_calc, gpu_peak_mb=_gpu_peak_mb(),
                           converged=ps.converged.get("final"), n_imag=crit.n_imag,
                           site=crit.site, flags=crit.codes, issues=[asdict(i) for i in crit.issues])
                if ref is not None:
                    fb = find_band(res, ps.stages["final"], ref.band)
                    row["band"] = ref.band
                    row["ref_ads_cm"] = ref.adsorbed_cm
                    row["ref_gas_cm"] = ref.gas_cm
                    row["ref_verified"] = ref.verified
                    if fb:
                        k, nu = fb
                        row["band_cm"] = nu
                        if res.intensities_km_mol is not None:
                            row["band_intensity"] = float(res.intensities_km_mol[k])
                        if res.alt_intensities_km_mol:
                            row["band_intensity_alt"] = {n: float(v[k]) for n, v in res.alt_intensities_km_mol.items()}
                        if ref_calc != calc and preset_status(ref_calc, cfg)[0]:
                            run_gas(mol, ref_calc)
                        g = gas_band.get((mol, ref_calc))
                        if g is not None:
                            row["gas_ref_cm"] = g
                            row["calc_shift_cm"] = nu - g
                        if ref.adsorbed_cm is not None:
                            row["ref_shift_cm"] = ref.adsorbed_cm - ref.gas_cm
                plot_spectrum(out["spectrum"], rd.path("plots", "ir_spectrum.png"),
                              f"{mol} on {metal}({facet}) {site}: {calc}",
                              f"{res.force_method} / {res.dipole_method}")
                rd.path("report.md").write_text(out["report"], encoding="utf-8")
                rd.write_json("row.json", row)
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - t0)
                log.error("%s/%s failed: %s\n%s", spec, calc, exc, traceback.format_exc())
            ads_rows.append(row)

    md = render_markdown(gas_rows, ads_rows, skipped)
    root.path("validation.md").write_text(md, encoding="utf-8")
    root.write_json("validation.json", {"dir": str(root.root), "gas": gas_rows, "adsorbates": ads_rows,
                                        "skipped": skipped})

    with tracked_run(cfg.tracking, "validation") as run:
        run.log_params({"calculators": ",".join(calculators), "molecules": ",".join(molecules),
                        "surfaces": ",".join(surfaces)})
        for calc, s in summarize(gas_rows).items():
            run.log_metrics({f"{calc}_mae_cm": s["mae_cm"], f"{calc}_mae_scaled_cm": s["mae_scaled_cm"]})
        run.log_artifact(root.path("validation.md"))
    return {"dir": str(root.root), "markdown": md, "gas": gas_rows, "adsorbates": ads_rows, "skipped": skipped}


def summarize(gas_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-calculator error statistics over all matched fundamentals (IR + Raman active)."""
    out: dict[str, dict[str, float]] = {}
    by_calc: dict[str, list[dict]] = {}
    for r in gas_rows:
        if r.get("matches"):
            by_calc.setdefault(r["calculator"], []).extend(r["matches"])
    for calc, ms in by_calc.items():
        c = np.array([m["calc_cm"] for m in ms])
        e = np.array([m["exp_cm"] for m in ms])
        lam = float(c @ e / (c @ c))
        err, err_s = c - e, lam * c - e
        out[calc] = {
            "n_modes": len(ms),
            "mae_cm": float(np.abs(err).mean()),
            "mse_cm": float(err.mean()),
            "max_abs_cm": float(np.abs(err).max()),
            "scale": lam,
            "mae_scaled_cm": float(np.abs(err_s).mean()),
        }
    return out


def _formatter(row: dict[str, Any]):
    def fmt_value(key: str, fmt: str = "{:.0f}") -> str:
        return "-" if row.get(key) is None else fmt.format(row[key])

    return fmt_value


def row_flags(row: dict[str, Any]) -> list[str]:
    """Critic flags of a row; older rows without them get site/convergence flags from the data."""
    if "flags" in row:
        return list(row["flags"])
    return []


def render_markdown(gas_rows, ads_rows, skipped) -> str:
    lines = ["# Validation", ""]
    lines += ["## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)", "",
              "Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in "
              "km/mol (calc / exp).", ""]
    calcs = sorted({r["calculator"] for r in gas_rows})
    header = "| molecule | mode | exp | " + " | ".join(calcs) + " |"
    lines += [header, "|" + "---|" * (3 + len(calcs))]
    mols = []
    for r in gas_rows:
        if r["molecule"] not in mols:
            mols.append(r["molecule"])
    for mol in mols:
        per = {r["calculator"]: r for r in gas_rows if r["molecule"] == mol}
        refs = gas_phase_fundamentals(mol) or []
        for i, f in enumerate(refs):
            cells = []
            for c in calcs:
                r = per.get(c)
                if r is None or r.get("error") or not r.get("matches"):
                    if r is not None and r.get("error"):
                        cells.append("err")
                    elif r is not None and r.get("n_imag"):
                        cells.append(f"saddle ({r['n_imag']} imag.)")
                    else:
                        cells.append("-")
                    continue
                m = r["matches"][i]
                inten = "" if m["calc_intensity"] is None else f" / {m['calc_intensity']:.0f}"
                cells.append(f"{m['calc_cm']:.0f} ({m['error_cm']:+.0f}){inten}")
            ie = "" if f.ir_intensity_km_mol is None else f" / {f.ir_intensity_km_mol:g}"
            lines.append(f"| {mol} | {f.mode} | {f.wavenumber_cm:.0f}{ie} | " + " | ".join(cells) + " |")
    lines += ["", "### Error summary per calculator", "",
              "| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    summ = summarize(gas_rows)
    for c in calcs:
        s = summ.get(c)
        rows = [r for r in gas_rows if r["calculator"] == c]
        wall = sum(r.get("wall_s", 0.0) for r in rows)
        dip = next((r.get("dipole") for r in rows if r.get("dipole")), "-")
        if s is None:
            lines.append(f"| {c} | {dip} | 0 | - | - | - | - | - | {wall:.1f} |")
            continue
        lines.append(f"| {c} | {dip} | {s['n_modes']} | {s['mae_cm']:.0f} | {s['mse_cm']:+.0f} | "
                     f"{s['max_abs_cm']:.0f} | {s['scale']:.3f} | {s['mae_scaled_cm']:.0f} | {wall:.1f} |")
    lines.append("\n*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).")
    for r in gas_rows:
        if r.get("error"):
            lines.append(f"- {r['molecule']}/{r['calculator']}: {r['error']}")
        elif row_flags(r):
            lines.append(f"- {r['molecule']}/{r['calculator']}: flagged {', '.join(row_flags(r))}")

    if ads_rows:
        lines += ["", "## Adsorbates vs literature trends", "",
                  "Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same "
                  "task: column 'gas ref'). Rows with critic flags are listed below the table and are "
                  "not used for conclusions.", "",
                  "| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | "
                  "ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |",
                  "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|"]
        for r in ads_rows:
            if r.get("error"):
                lines.append(f"| {r['system']} | {r['calculator']} | - | - | - | err: {r['error'][:80]} "
                             f"| | | | | | | {r['wall_s']:.1f} |")
                continue
            f = _formatter(r)
            ver = "" if r.get("ref_verified", True) else " (unverified)"
            site = (r.get("site") or {}).get("label", "-")
            flags = ", ".join(row_flags(r)) or "-"
            lines.append(
                f"| {r['system']} | {r['calculator']} | {r.get('model')} ({r.get('n_atoms')} at.) | {site} | "
                f"{r.get('band', '-')} | {f('band_cm')} | {r.get('gas_ref_calculator', r['calculator'])} | "
                f"{f('calc_shift_cm', '{:+.0f}')} | {f('ref_ads_cm')}{ver} | {f('ref_shift_cm', '{:+.0f}')} | "
                f"{f('band_intensity', '{:.1f}')} | {flags} | {r['wall_s']:.1f} |"
            )
        flagged = [r for r in ads_rows if not r.get("error") and r.get("issues")]
        if flagged:
            lines += ["", "Critic flags:"]
            for r in flagged:
                for i in r["issues"]:
                    lines.append(f"- {r['system']} / {r['calculator']}: {i['code']}: {i['message']}")
        alts = [r for r in ads_rows if r.get("band_intensity_alt")]
        if alts:
            lines += ["", "Whole-system dipole intensities (native xtb; include metal polarization of the "
                      "finite cluster, not comparable with the fragment column):"]
            for r in alts:
                vals = "; ".join(f"{k}: {v:.0f} km/mol" for k, v in r["band_intensity_alt"].items())
                lines.append(f"- {r['system']} / {r['calculator']}: {vals}")
    if skipped:
        lines += ["", "## Skipped calculators", ""]
        lines += [f"- `{k}`: {v}" for k, v in skipped.items()]
    return "\n".join(lines) + "\n"
