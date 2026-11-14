"""Run C: adsorbate/surface systems from semiconductor processing.

* Metal part (interconnect / liner): CO atop on Cu(111) and Ru(0001). This goes through
  :func:`~ir_agent.validation.run.run_validation`, so each shift is taken against a same-task gas
  reference and the critic flags are applied, exactly as in Run A. Clusters use the NN-scaled cut-off
  (``geometry.cluster_radius_nn``), which gives Cu10/Ru10 like Pt10/Au10.
* Si/SiO2 part: H/Si(100)-2x1, NH2+H/Si(100)-2x1 and an isolated silanol (:mod:`ir_agent.semis`).
  These are compared as *absolute* frequencies with the measured bands in
  ``data/reference/experimental_surface_ir.csv``. The measured-vs-harmonic gap from overtone data
  (``anharmonicity.csv``) is reported next to them rather than folded into a scale factor.

Hessians are partial (adsorbate + top layer). Cluster rows get GFN2-xTB whole-cluster dipoles.
Si slab rows are frequencies only: a semiconductor has no metal-surface selection rule, and an
adsorbate-only fragment dipole of H atoms is meaningless.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import write

from ir_agent.calculators.registry import (
    PRESETS,
    canonical_preset,
    make_calculator,
    make_dipole_evaluator,
    make_force_evaluator,
    preset_status,
)
from ir_agent.config import AppConfig
from ir_agent.critic import Critique, Issue
from ir_agent.pipeline import relax
from ir_agent.runtime import RunDir, get_logger
from ir_agent.semis import SYSTEMS, assign_bands, broken_attachments, build
from ir_agent.validation.experimental import anharmonic_gap, comparison_value
from ir_agent.validation.run import _gpu_peak_mb, _load_row, _sub, run_validation, uma_task
from ir_agent.vibrations.ir import compute_ir

log = get_logger("run_c")

METAL_SURFACES = ["CO/Cu/111", "CO/Ru/0001"]
CALCULATORS = ["xtb-gfn1", "uma-omol", "uma-oc20", "uma-omat"]
CLUSTER_RADIUS_NN = 1.083  #: 3.0 A for Pt -> Pt10/Au10/Cu10/Ru10


def run_c_config(cfg: AppConfig) -> AppConfig:
    g = cfg.geometry.model_copy(update={"cluster_radius_nn": CLUSTER_RADIUS_NN})
    return cfg.model_copy(update={"geometry": g})


def model_for(calc: str) -> str:
    return "slab" if PRESETS[calc].periodic else "cluster"


def _critique(sysm, res, converged: bool, steps: int, cfg: AppConfig) -> Critique:
    c = cfg.critic
    out = Critique()
    freqs = np.asarray(res.frequencies_cm, float)
    out.lowest_cm = float(freqs.min()) if freqs.size else None
    out.n_imag = int((freqs < -c.imag_threshold_cm).sum())
    if not converged:
        out.issues.append(Issue("not_converged", "flag",
                                f"final optimization stopped before fmax={cfg.optimization.fmax_final} eV/A "
                                f"({steps} steps)"))
    if out.n_imag:
        out.issues.append(Issue("imaginary_modes", "flag",
                                f"{out.n_imag} imaginary mode(s) below -{c.imag_threshold_cm:g} cm^-1 "
                                f"(lowest {out.lowest_cm:.0f} cm^-1)"))
    broken = broken_attachments(sysm)
    if broken:
        out.issues.append(Issue("site_changed", "flag", "bond(s) broken or H hopped: " + "; ".join(broken)))
    return out


def _compare(system: str, bands: list[dict]) -> list[dict]:
    out = []
    for b in bands:
        r = dict(b)
        ref = comparison_value(system, b["band"])
        if ref is not None and ref["value"] is not None:
            r.update(exp_cm=ref["value"], exp_verified=ref["verified_bool"], exp_technique=ref.get("technique"),
                     exp_coverage=ref.get("coverage"), exp_T_K=ref.get("temperature_K"),
                     exp_citation=ref.get("citation"), exp_doi=ref.get("doi"),
                     calc_minus_exp=b["freq_cm"] - ref["value"])
            gap = anharmonic_gap(system, b["band"])
            if gap is not None and gap["gap"] is not None:
                r.update(anh_gap_cm=gap["gap"], anh_verified=gap["verified_bool"], anh_source=gap.get("citation"),
                         harmonic_est_cm=ref["value"] + gap["gap"],
                         calc_minus_harmonic_est=b["freq_cm"] - ref["value"] - gap["gap"])
        out.append(r)
    return out


def run_semis_case(name: str, calc: str, cfg: AppConfig, rd: RunDir) -> dict[str, Any]:
    model = model_for(calc)
    sysm = build(name, model)
    atoms = sysm.atoms
    kind = "surface" if model == "slab" else "cluster"
    row: dict[str, Any] = {"system": name, "calculator": calc, "model": model, "formula": sysm.formula,
                           "n_atoms": len(atoms), "task": uma_task(calc, kind, cfg),
                           "hessian_atoms": len(sysm.hessian_indices)}
    write(rd.path("structures", "initial.xyz"), atoms)
    t0 = time.perf_counter()
    wd = rd.root / "calc"
    c_opt = make_calculator(calc, cfg, atoms=atoms, system_kind=kind, charge=0, spin=1, workdir=wd)
    atoms, n_pre, conv_pre = relax(atoms, c_opt, cfg.optimization.fmax_preopt, cfg, "preopt", rd)
    c_freq = make_calculator(calc, cfg, atoms=atoms, system_kind=kind, mode="freq", charge=0, spin=1, workdir=wd)
    atoms, n_fin, conv_fin = relax(atoms, c_freq, cfg.optimization.fmax_final, cfg, "final", rd)
    retry = None
    if not conv_fin and cfg.critic.enabled and cfg.critic.max_retries > 0:
        # the critic's not_converged retry (as in the agent graph): FIRE with twice the step budget.
        # BFGS overshoots on very soft modes (e.g. the NH2 wag/rotation on Si) and stalls.
        o = cfg.optimization
        opt = o.model_copy(update={"optimizer": cfg.critic.retry_optimizer, "max_steps": 2 * o.max_steps,
                                   "stall_steps": 2 * o.stall_steps})
        rcfg = cfg.model_copy(update={"optimization": opt})
        log.info("%s/%s not converged after %d steps: critic retry with %s", name, calc, n_fin, opt.optimizer)
        atoms, n_retry, conv_fin = relax(atoms, c_freq, o.fmax_final, rcfg, "final_retry", rd)
        retry = {"optimizer": opt.optimizer, "steps": n_retry, "converged": conv_fin, "first_steps": n_fin}
        n_fin += n_retry
    sysm.atoms = atoms
    write(rd.path("structures", "final.xyz"), atoms)
    t_opt = time.perf_counter() - t0

    de = None
    if model == "cluster" or name == "SiOH/SiO2":
        # whole cluster (the silanol cluster is molecule-like also in its periodic box)
        de = make_dipole_evaluator("xtb", cfg, charge=0, spin=1, region=list(range(len(atoms))), workdir=wd)
        de.name = "GFN2-xTB (xtb), whole cluster"
    idx = sysm.hessian_indices
    t1 = time.perf_counter()
    if calc in ("xtb", "xtb-gfn1"):
        hess = c_freq.hessian(atoms)
        res = compute_ir(atoms, None, de, indices=idx, delta=cfg.vibrations.delta, nfree=cfg.vibrations.nfree,
                         hessian=hess, hessian_label=f"{c_freq.method_label} (xtb --hess)")
    else:
        fe = make_force_evaluator(calc, cfg, atoms=atoms, system_kind=kind, charge=0, spin=1, workdir=wd)
        res = compute_ir(atoms, fe, de, indices=idx, delta=cfg.vibrations.delta, nfree=cfg.vibrations.nfree)
    t_hess = time.perf_counter() - t1
    crit = _critique(sysm, res, conv_fin, n_fin, cfg)
    bands = assign_bands(res, atoms, sysm)  # compared with experiment at render time
    row.update(wall_s=time.perf_counter() - t0, opt_s=t_opt, hessian_s=t_hess,
               opt_steps={"preopt": n_pre, "final": n_fin}, converged={"preopt": conv_pre, "final": conv_fin},
               critic_retry=retry,
               force_method=res.force_method, dipole=res.dipole_method, n_disp=res.n_displacements,
               frequencies_cm=[float(f) for f in res.frequencies_cm],
               intensities_km_mol=None if res.intensities_km_mol is None else [float(x) for x in res.intensities_km_mol],
               bands=bands, n_imag=crit.n_imag, lowest_cm=crit.lowest_cm, flags=crit.codes,
               issues=[asdict(i) for i in crit.issues], gpu_peak_mb=_gpu_peak_mb())
    return row


def run_semis(cfg: AppConfig, calculators: list[str], root: RunDir,
              systems: tuple[str, ...] = SYSTEMS) -> list[dict[str, Any]]:
    rows = []
    for calc_name in calculators:
        calc = canonical_preset(calc_name)
        ok, reason = preset_status(calc, cfg)
        if not ok:
            log.warning("Skipping %s: %s", calc, reason)
            rows.append({"calculator": calc, "error": f"unavailable: {reason}"})
            continue
        for name in systems:
            tag = name.replace("/", "_").replace("+", "").replace("(", "").replace(")", "")
            rd = _sub(root, f"semi_{tag}_{calc}")
            done = _load_row(rd)
            if done is not None:
                rows.append(done)
                continue
            t0 = time.perf_counter()
            try:
                row = run_semis_case(name, calc, cfg, rd)
                rd.write_json("row.json", row)
            except Exception as exc:
                row = {"system": name, "calculator": calc, "error": f"{type(exc).__name__}: {exc}",
                       "wall_s": time.perf_counter() - t0}
                log.error("%s/%s failed: %s\n%s", name, calc, exc, traceback.format_exc())
            rows.append(row)
    return rows


def render_semis_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["## Si / SiO2 systems: absolute wavenumbers vs experiment (cm^-1)", "",
             "calc = harmonic (partial Hessian: adsorbate + top layer). exp = measured fundamental "
             "(data/reference/experimental_surface_ir.csv; '(unverified)' = not read in the primary "
             "paper). gap = harmonic minus fundamental from overtone data (anharmonicity.csv); "
             "harm. est. = exp + gap.", "",
             "| system | calculator | model | band | calc | exp | calc - exp | gap | harm. est. | calc - harm. est. | I (km/mol) | flags |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r.get('system', '-')} | {r['calculator']} | - | - | - | - | - | - | - | - | - | error: {r['error'][:60]} |")
            continue
        flags = ", ".join(r.get("flags") or []) or "-"
        for b in _compare(r["system"], r.get("bands") or [{"band": "(no band found)", "freq_cm": None}]):
            def f(v, fmt="{:.0f}"):
                return "-" if v is None else fmt.format(v)
            exp = "-" if b.get("exp_cm") is None else (f(b["exp_cm"]) + ("" if b.get("exp_verified") else " (unverified)"))
            inten = b.get("intensity_km_mol")
            lines.append(
                f"| {r['system']} | {r['calculator']} | {r['model']} ({r['formula']}) | {b['band']} | {f(b.get('freq_cm'))} | "
                f"{exp} | {f(b.get('calc_minus_exp'), '{:+.0f}')} | {f(b.get('anh_gap_cm'), '{:+.0f}')} | "
                f"{f(b.get('harmonic_est_cm'))} | {f(b.get('calc_minus_harmonic_est'), '{:+.0f}')} | "
                f"{'n/a' if inten is None else f'{inten:.0f}'} | {flags} |")
    issues = [(r, i) for r in rows for i in r.get("issues") or []]
    if issues:
        lines += ["", "Critic flags:"]
        lines += [f"- {r['system']} / {r['calculator']}: {i['code']}: {i['message']}" for r, i in issues]
    return "\n".join(lines) + "\n"


def run_c(cfg: AppConfig, calculators: list[str], resume_dir: str | Path | None = None,
          parts: tuple[str, ...] = ("metal", "semis")) -> dict[str, Any]:
    cfg = run_c_config(cfg)
    root = RunDir(Path(resume_dir)) if resume_dir else RunDir.create(cfg.run.output_root, "run_c")
    out: dict[str, Any] = {"dir": str(root.root), "calculators": calculators}
    md = [f"# Run C ({', '.join(calculators)})", ""]
    if "metal" in parts:
        v = run_validation(cfg, calculators, ["CO"], METAL_SURFACES, resume_dir=root.root)
        out["metal"] = {"gas": v["gas"], "adsorbates": v["adsorbates"], "skipped": v["skipped"]}
        md += ["## CO on Cu(111) and Ru(0001)", "", "See `validation.md` in this directory.", ""]
    if "semis" in parts:
        rows = run_semis(cfg, calculators, root)
        out["semis"] = rows
        md.append(render_semis_markdown(rows))
    root.write_json("run_c.json", out)
    root.path("run_c.md").write_text("\n".join(md), encoding="utf-8")
    return out


def rerender(run_dir: str | Path) -> str:
    """Rebuild run_c.md from the saved rows (after the reference CSVs change)."""
    import json

    root = Path(run_dir)
    out = json.loads((root / "run_c.json").read_text(encoding="utf-8"))
    md = [f"# Run C ({', '.join(out['calculators'])})", ""]
    if "metal" in out:
        md += ["## CO on Cu(111) and Ru(0001)", "", "See `validation.md` in this directory.", ""]
    if "semis" in out:
        md.append(render_semis_markdown(out["semis"]))
    text = "\n".join(md)
    (root / "run_c.md").write_text(text, encoding="utf-8")
    return text
