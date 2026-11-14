"""Relative CO site energies on Pt(111) from the Run B relaxed structures (same UMA task, same cell).

    python scripts/site_energies.py                     # CUDA (optional GPU guard, --gb 3.0)
    python scripts/site_energies.py --inner --device cpu

For each Run B point (task x cell, delta 0.01), single-point energies of the final CO/Pt structures
(ontop, bridge, fcc) with the same task. Same composition and cell, so E(site) - E(ontop) is the
relative adsorption energy. Experiment puts CO atop at low coverage on Pt(111); GGA-level DFT
(PBE/RPBE, the levels behind omat/oc20) is known to prefer the hollow sites ("CO/Pt(111) puzzle").
The relaxed site (critic classifier) is reported next to the requested one, so hops are visible.
Writes runs/uma_run_b/site_energies.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_agent.gpu import guarded  # noqa: E402  (stdlib-only module)

RUN_B = ROOT / "runs" / "uma_run_b"
PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
SITES = ("ontop", "bridge", "fcc")


def inner(device: str) -> int:
    from ase.io import read

    from ir_agent.calculators.registry import make_calculator
    from ir_agent.config import load_config
    from ir_agent.gpu import apply_gpu_guard_cap
    from ir_agent.sites import classify_site

    apply_gpu_guard_cap()
    cfg = load_config(overrides={"calculators": {"uma": {"device": device}}, "tracking": {"mlflow": False}})
    rows = []
    for point in sorted(RUN_B.glob("uma-*_d0.01")):
        preset = point.name.split("_")[0]
        val = point / "validation"
        energies = {}
        for site in SITES:
            tag = "" if site == "ontop" else f"_{site}"
            f = val / f"ads_CO_Pt111{tag}_{preset}" / "structures" / "final.xyz"
            if not f.exists():
                continue
            atoms = read(f)
            calc = make_calculator(preset, cfg, atoms=atoms, system_kind="surface")
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            n_metal = sum(1 for s in atoms.get_chemical_symbols() if s == "Pt")
            ads = list(range(n_metal, len(atoms)))
            info = classify_site(atoms, ads)
            energies[site] = {"E_eV": e, "relaxed": info.label}
        if "ontop" in energies:
            e0 = energies["ontop"]["E_eV"]
            for v in energies.values():
                v["dE_vs_ontop_eV"] = v["E_eV"] - e0
        rows.append({"point": point.name, "preset": preset, "sites": energies})
    (RUN_B / "site_energies.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("| point | " + " | ".join(f"{s}: dE (eV), relaxed site" for s in SITES) + " |")
    print("|---|" + "---|" * len(SITES))
    for r in rows:
        cells = []
        for s in SITES:
            v = r["sites"].get(s)
            cells.append("-" if v is None else f"{v.get('dE_vs_ontop_eV', float('nan')):+.3f}, {v['relaxed']}")
        print(f"| {r['point']} | " + " | ".join(cells) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gb", default="3.0")
    args = ap.parse_args()
    if args.inner:
        return inner(args.device)
    cmd = guarded([str(PY), str(Path(__file__).resolve()), "--inner", "--device", "cuda"],
                  args.gb, "Run B: CO/Pt site single-point energies")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
