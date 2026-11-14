"""Regenerate the REPORT/README summary figures from saved run outputs (no new calculations).

Inputs (tracked copies of run outputs):
  docs/validation/validation_xtb.json   <- GFN2/GFN1-xTB (xtb --hess route)
  docs/validation/validation_dxtb.json  <- dxtb (GFN1-xTB in PyTorch)
  docs/validation/validation_orb.json   <- Orb presets
  docs/validation/validation_pet.json   <- PET presets
  docs/validation/validation_uma.json   <- UMA-S 1.2.1 tasks
  docs/validation/adsorbate_sites.json  (binding site per case, from the relaxed geometries)
  docs/benchmarks/uma_proxy_benchmark_guarded.json
Outputs: docs/figures/{gas_phase_errors,gas_mae_summary,adsorbate_shifts,uma_proxy_benchmark}.png

Palette: categorical slots of the dataviz reference palette in fixed order, validated (scatter:
first 3 slots all-pairs; bars: adjacent pairs). Marker shapes / direct value labels are the
secondary encoding, and every figure has a numeric table in REPORT.md.

Usage:  python scripts/make_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
VAL = ROOT / "docs" / "validation"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
REF_GRAY = "#8f8d86"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NAMES = {"orb-mol": "OrbMol-v2", "pet-omol": "PET-OMol-L", "orb-omat": "Orb-v3-OMat",
         "pet-mad": "PET-MAD-1.6-S", "xtb": "GFN2-xTB", "xtb-gfn1": "GFN1-xTB (xtb)",
         "dxtb": "GFN1-xTB (dxtb)", "uma-omol": "UMA-S 1.2.1 omol", "uma-oc20": "UMA-S 1.2.1 oc20",
         "uma-omat": "UMA-S 1.2.1 omat"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})


#: (file, run dir if the JSON predates the "dir" field, calculators taken from it).
SOURCES = [
    ("validation_xtb.json", None, {"xtb", "xtb-gfn1"}),
    ("validation_orb.json", "runs/20261113-020635_validation", {"orb-mol", "orb-omat"}),
    ("validation_pet.json", "runs/20261113-020717_validation", {"pet-omol", "pet-mad"}),
    ("validation_dxtb.json", "runs/20261113-003214_validation", {"dxtb"}),
    ("validation_uma.json", None, {"uma-omol", "uma-oc20", "uma-omat"}),
]


def _rel_dir(d: dict, fallback: str | None) -> str | None:
    if d.get("dir"):
        p = Path(d["dir"])
        return (p.relative_to(ROOT) if p.is_absolute() else p).as_posix()
    return fallback


def _ads_subrun(system: str, calc: str) -> str:
    mol, rest = system.split("/", 1)
    metal, facet = rest.split("(")[0], rest.split("(")[1].split(")")[0]
    site = rest.split("/")[1] if "/" in rest else None
    return f"ads_{mol}_{metal}{facet}{'_' + site if site else ''}_{calc}"


def load_rows() -> tuple[dict, dict]:
    """Gas and adsorbate rows per (molecule|system, calculator), with critic-style flags added to
    rows that predate the critic (from adsorbate_sites.json and convergence.json)."""
    sites = json.loads((VAL / "adsorbate_sites.json").read_text(encoding="utf-8"))
    conv = json.loads((VAL / "convergence.json").read_text(encoding="utf-8"))
    gas, ads = {}, {}
    for fname, fallback, calcs in SOURCES:
        if not (VAL / fname).exists():
            continue
        d = json.loads((VAL / fname).read_text(encoding="utf-8"))
        rdir = _rel_dir(d, fallback)
        for r in d["gas"]:
            if r["calculator"] in calcs:
                r = dict(r)
                c = conv.get(f"{rdir}/gas_{r['molecule']}_{r['calculator']}", {})
                r.setdefault("converged", c.get("converged"))
                gas[(r["molecule"], r["calculator"])] = r
        for r in d["adsorbates"]:
            if r["calculator"] not in calcs:
                continue
            r = dict(r)
            if "flags" not in r and not r.get("error"):
                flags = []
                c = conv.get(f"{rdir}/{_ads_subrun(r['system'], r['calculator'])}", {})
                if c.get("converged") is False:
                    flags.append("not_converged")
                s = sites.get(f"{r['system']}|{r['calculator']}", {})
                if s.get("site") in ("physisorbed", "desorbed"):
                    flags.append("weakly_bound")
                elif s.get("site_changed"):
                    flags.append("site_changed")
                r["flags"], r["site"] = flags, s or None
            mol = r["system"].split("/")[0]
            g = gas.get((mol, r.get("gas_ref_calculator", r["calculator"])), {})
            if g.get("converged") is False:
                r["flags"] = [*r.get("flags", []), "gas_ref_not_converged"]
            ads[(r["system"], r["calculator"])] = r
    return gas, ads


FLAG_NOTES = {"site_changed": "site changed", "weakly_bound": "physisorbed", "not_converged": "not converged",
              "imaginary_modes": "imag. modes", "gas_ref_not_converged": "gas ref not conv."}


def gas_phase_errors(gas: dict) -> Path:
    series = [("xtb", SLOT[0], "o"), ("orb-mol", SLOT[1], "s"), ("pet-omol", SLOT[2], "^")]
    dodge = {"xtb": -40.0, "orb-mol": 0.0, "pet-omol": 40.0}
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.axhline(0, color=INK2, lw=1.0)
    for calc, color, marker in series:
        xs, ys = [], []
        for (_mol, c), r in gas.items():
            if c != calc or not r.get("matches"):
                continue
            for m in r["matches"]:
                xs.append(m["exp_cm"] + dodge[calc])
                ys.append(m["calc_cm"] - m["exp_cm"])
        ax.scatter(xs, ys, s=46, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.6,
                   label=NAMES[calc], zorder=3)
    g2 = [(m["exp_cm"], m["calc_cm"] - m["exp_cm"], f"{mol} {m['mode']}")
          for (mol, c), r in gas.items() if c == "xtb" and r.get("matches") for m in r["matches"]]
    for x, y, n in sorted(g2, key=lambda t: -abs(t[1]))[:3]:
        ax.annotate("GFN2: " + n.replace("sym. deformation (umbrella)", "umbrella"), (x + dodge["xtb"], y),
                    xytext=(10, 0) if x < 1200 else (-10, 0), textcoords="offset points", fontsize=8.5,
                    color=INK2, ha="left" if x < 1200 else "right", va="center")
    ax.set_xlabel("Experimental fundamental (cm$^{-1}$, NIST CCCBDB)")
    ax.set_ylabel("Harmonic calc. minus exp. (cm$^{-1}$)")
    ax.set_title("Gas-phase fundamentals of H$_2$O, CO, NH$_3$, CO$_2$: error per mode", loc="left",
                 fontsize=11, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3, fontsize=9, borderaxespad=0.2)
    ax.set_ylim(-250, 300)
    fig.tight_layout()
    path = OUT / "gas_phase_errors.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def gas_mae_summary(gas: dict) -> Path:
    order = ["uma-omol", "uma-oc20", "uma-omat", "orb-mol", "pet-omol", "orb-omat", "pet-mad", "dxtb", "xtb",
             "xtb-gfn1"]
    order = [c for c in order if any(cc == c and r.get("matches") for (_m, cc), r in gas.items())]
    mae, mae_s = [], []
    for c in order:
        ms = [m for (_mol, cc), r in gas.items() if cc == c and r.get("matches") for m in r["matches"]]
        calc = np.array([m["calc_cm"] for m in ms])
        exp = np.array([m["exp_cm"] for m in ms])
        lam = float(calc @ exp / (calc @ calc))
        mae.append(float(np.abs(calc - exp).mean()))
        mae_s.append(float(np.abs(lam * calc - exp).mean()))
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    h = 0.36
    b1 = ax.barh(y + h / 2, mae, height=h, color=SLOT[0], edgecolor=SURFACE, linewidth=2,
                 label="MAE, harmonic as computed")
    b2 = ax.barh(y - h / 2, mae_s, height=h, color=SLOT[1], edgecolor=SURFACE, linewidth=2,
                 label="MAE after one in-sample scale factor")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [NAMES[c] for c in order])
    ax.set_xlabel("Mean absolute error vs NIST fundamentals, 11 modes (cm$^{-1}$)")
    ax.set_title("Gas-phase frequency error by method", loc="left", fontsize=11, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, fontsize=9, borderaxespad=0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(mae) * 1.15)
    fig.tight_layout()
    path = OUT / "gas_mae_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _shift_panel(ax, ads: dict, series: list[tuple[str, str]], systems: list[str], title: str,
                 notes: list[str]) -> None:
    """Grouped bars; flagged bars are hatched and get a footnote letter (text collected in ``notes``)."""
    from matplotlib.patches import Patch

    from ir_agent.validation.reference import ADSORBATE_REFERENCES

    n = len(series) + 1
    width = 0.8 / n
    ax.axhline(0, color=INK2, lw=1.0)
    for j, (key, color) in enumerate([*series, ("ref", REF_GRAY)]):
        for i, sysname in enumerate(systems):
            x = i - 0.4 + width * (j + 0.5)
            hatch, note, empty = None, None, None
            if key == "ref":
                mol, metal = sysname.split("/")[0], sysname.split("/")[1].split("(")[0]
                ref = ADSORBATE_REFERENCES.get((mol, metal))
                v = None if ref is None or ref.adsorbed_cm is None else ref.adsorbed_cm - ref.gas_cm
                if v is None:
                    empty = "no value"
                else:
                    if not ref.verified:
                        note = f"{sysname} reference: experimental, not verified"
                    if ref.shift_window:
                        lo, hi = ref.shift_window
                        ax.plot([x, x], [lo, hi], color=INK2, lw=1.2, zorder=4)
                        for yy in (lo, hi):
                            ax.plot([x - width * 0.25, x + width * 0.25], [yy, yy], color=INK2, lw=1.2, zorder=4)
            else:
                r = ads.get((sysname, key), {})
                v = r.get("calc_shift_cm")
                if r.get("error"):
                    empty = "SCC not conv." if "SCC" in r["error"] else "SCF diverged"
                elif r.get("flags"):
                    hatch = "////"
                    note = f"{NAMES[key]} {sysname}: " + ", ".join(FLAG_NOTES.get(f, f) for f in r["flags"])
            if v is None:
                ax.text(x, 8, empty or "n/a", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
                continue
            ax.bar(x, v, width=width * 0.92, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, hatch=hatch)
            tag = ""
            if note:
                notes.append(note)
                tag = chr(ord("a") + len(notes) - 1)
            ax.text(x, v + (8 if v >= 0 else -8), f"{v:+.0f}" + (f"$^{tag}$" if tag else ""), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    labels = [s.replace("NH3", "NH$_3$") + ("\nC-O stretch" if s.startswith("CO") else "\numbrella") for s in systems]
    ax.set_xticks(range(len(systems)), labels, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10.5)
    handles = [Patch(facecolor=c, label=NAMES[k]) for k, c in series]
    handles.append(Patch(facecolor=REF_GRAY, label="Literature (whisker: consistent range)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncols=1)


def adsorbate_shifts(ads: dict) -> Path:
    systems = ["CO/Pt(111)", "NH3/Pt(111)", "CO/Au(111)", "NH3/Au(111)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    notes: list[str] = []
    _shift_panel(a1, ads, [("xtb", SLOT[0]), ("xtb-gfn1", SLOT[1]), ("orb-mol", SLOT[2]), ("pet-omol", SLOT[3])],
                 systems, "Pt$_{10}$/Au$_{10}$ cluster models (non-periodic)", notes)
    _shift_panel(a2, ads, [("orb-omat", SLOT[4]), ("pet-mad", SLOT[5])], systems,
                 "Periodic 3x3x3 slabs (materials MLIPs)", notes)
    a1.set_ylabel("Band shift on adsorption (cm$^{-1}$)\nadsorbed minus same-method gas")
    a1.set_ylim(-500, 300)
    fig.suptitle("Adsorption-induced band shifts vs literature (hatched = flagged by the critic, not used "
                 "for conclusions)", x=0.01, ha="left", fontsize=11)
    foot = "   ".join(f"{chr(ord('a') + k)}: {t}" for k, t in enumerate(notes))
    fig.text(0.01, 0.01, foot.replace("NH3", "NH$_3$"), fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUT / "adsorbate_shifts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def uma_proxy(bench: dict) -> Path:
    systems = [("H2O gas (omol)", "H$_2$O (3 atoms)"), ("NH3 gas (omol)", "NH$_3$ (4 atoms)"),
               ("CO/Pt(111) 3x3x3 (oc20)", "CO/Pt(111) slab (29 atoms)")]
    markers = ["o", "s", "^"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for (key, label), c, mk in zip(systems, SLOT[:3], markers, strict=True):
        rows = sorted((r for r in bench["rows"] if r["system"] == key), key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in rows]
        a1.plot(bs, [r["per_image_ms"] for r in rows], color=c, lw=2, marker=mk, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        a2.plot(bs, [r["peak_vram_mb"] / 1024 for r in rows], color=c, lw=2, marker=mk, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    cap = bench["meta"].get("gpu_guard_gb") or 3.5
    a2.axhline(cap, color=INK2, ls="--", lw=1)
    a2.text(24, cap + 0.06, f"PyTorch memory cap {cap:g} GB", va="bottom", ha="right", fontsize=8.5, color=INK2)
    a1.set_xlabel("Displaced geometries per forward pass")
    a1.set_ylabel("Wall time per displaced geometry (ms)")
    a1.set_title("Throughput", loc="left", fontsize=10.5)
    a2.set_xlabel("Displaced geometries per forward pass")
    a2.set_ylabel("Peak allocated VRAM (GiB)")
    a2.set_title("Memory", loc="left", fontsize=10.5)
    a2.set_ylim(0, 4)
    for a in (a1, a2):
        a.set_xticks([1, 6, 12, 24])
    a1.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Batched finite-difference Hessian, random-weight UMA-S proxy (146 M params), RTX 3070 Laptop",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path = OUT / "uma_proxy_benchmark.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _stats(ms: list[dict], key: str = "exp_cm") -> tuple[float, float]:
    calc = np.array([m["calc_cm"] for m in ms])
    exp = np.array([m[key] for m in ms])
    lam = float(calc @ exp / (calc @ calc))
    return float(np.abs(calc - exp).mean()), float(np.abs(lam * calc - exp).mean())


def write_tables(gas: dict, ads: dict) -> Path:
    """docs/validation/summary_tables.md: the numbers behind the figures (pasted into REPORT/README)."""
    from ir_agent.validation.reference import harmonic_reference

    calcs = [c for c in NAMES if any(cc == c for (_m, cc) in gas)]
    lines = ["# Summary tables (generated by scripts/make_report_figures.py)", "",
             "## Gas phase", "",
             "MAE over 11 fundamentals (NIST CCCBDB); 'scaled' = after one in-sample least-squares scale "
             "factor. 'Harmonic MAE' compares the 4 modes with experimental harmonic frequencies (H2O x3, "
             "CO; CCCBDB), i.e. like with like.", "",
             "| method | MAE | MAE scaled | harmonic MAE (4 modes) | mean signed vs harmonic |",
             "|---|---:|---:|---:|---:|"]
    for c in calcs:
        ms = [m for (_mol, cc), r in gas.items() if cc == c and r.get("matches") for m in r["matches"]]
        if not ms:
            continue
        mae, mae_s = _stats(ms)
        hm = [(m["calc_cm"], harmonic_reference(mol, m["mode"]))
              for (mol, cc), r in gas.items() if cc == c and r.get("matches") for m in r["matches"]]
        hm = [(a, b) for a, b in hm if b is not None]
        h_err = np.array([a - b for a, b in hm])
        lines.append(f"| {NAMES[c]} | {mae:.0f} | {mae_s:.0f} | {np.abs(h_err).mean():.0f} | {h_err.mean():+.0f} |")
    lines += ["", "## Adsorbate band shifts (cm^-1, adsorbed minus same-method, same-task gas band)", "",
              "| system | method | model | relaxed site | band | shift | reference | I fragment (km/mol) | flags |",
              "|---|---|---|---|---|---:|---|---:|---|"]
    from ir_agent.validation.reference import ADSORBATE_REFERENCES

    for (system, c), r in ads.items():
        if c not in NAMES:
            continue
        mol, metal = system.split("/")[0], system.split("/")[1].split("(")[0]
        ref = ADSORBATE_REFERENCES.get((mol, metal))
        if ref is None or ref.adsorbed_cm is None:
            rtxt = "trend only"
        else:
            rtxt = f"{ref.adsorbed_cm - ref.gas_cm:+.0f}" + ("" if ref.verified else " (unverified)")
            if ref.shift_window:
                rtxt += f", consistent {ref.shift_window[0]:+.0f}..{ref.shift_window[1]:+.0f}"
        if r.get("error"):
            lines.append(f"| {system} | {NAMES[c]} | - | - | - | - | {rtxt} | - | error: "
                         f"{'SCC not converged' if 'SCC' in r['error'] else r['error'][:40]} |")
            continue
        site = (r.get("site") or {}).get("label", "-")
        shift = "-" if r.get("calc_shift_cm") is None else f"{r['calc_shift_cm']:+.0f}"
        inten = "-" if r.get("band_intensity") is None else f"{r['band_intensity']:.0f}"
        if c == "dxtb":  # whole-cluster Mulliken-charge dipole: not comparable, not reported
            inten = "n/a"
        flags = ", ".join(FLAG_NOTES.get(f, f) for f in r.get("flags") or []) or "-"
        band = "C-O" if r.get("band", "").startswith("C-O") else "umbrella"
        lines.append(f"| {system} | {NAMES[c]} | {r.get('model')} | {site} | {band} | {shift} | {rtxt} | "
                     f"{inten} | {flags} |")
    path = VAL / "summary_tables.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gas, ads = load_rows()
    bench = json.loads((ROOT / "docs/benchmarks/uma_proxy_benchmark_guarded.json").read_text(encoding="utf-8"))
    for p in (gas_phase_errors(gas), gas_mae_summary(gas), adsorbate_shifts(ads), uma_proxy(bench),
              write_tables(gas, ads)):
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
