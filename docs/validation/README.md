# docs/validation: provenance

Each `*.json` / `*.md` pair is the output of one validation run (`ir-agent validate`, or the Run A/B/C
scripts), copied here from `runs/` (local, not published). The published copies are curated:
- only the rows that REPORT.md uses are kept (superseded or unreported rows are removed);
- the reference columns (`ref_*`) are filled from the current `src/ir_agent/validation/reference.py`;
- rows that predate the critic carry the convergence and site flags derived from
  `convergence.json` / `adsorbate_sites.json` (same logic as `scripts/make_report_figures.py`);
- run directories are stored as relative paths.

Computed values (frequencies, shifts, intensities, wall times) are unchanged.

| file | source run | rows |
|---|---|---|
| `validation_xtb.json/.md` | `runs/20261113-032834_validation` | GFN2-xTB, GFN1-xTB (`xtb --hess` route) |
| `validation_dxtb.json/.md` | `runs/20261113-003214_validation` | dxtb (GFN1-xTB in PyTorch) frequencies; its Mulliken-charge intensities are not reported |
| `validation_orb.json/.md` | `runs/20261113-020635_validation` | orb-mol, orb-omat |
| `validation_pet.json/.md` | `runs/20261113-020717_validation` | pet-omol, pet-mad |
| `validation_uma.json/.md` | `runs/20261113-105722_validation` (Run A, CUDA) | uma-omol, uma-oc20, uma-omat |
| `run_b/<point>.json/.md`, `run_b/summary.json`, `run_b/site_energies.json` | `runs/uma_run_b/` (Run B, CUDA) | uma-oc20, uma-omat: CO ontop/bridge/fcc and NH3 ontop, 3x3/4x4 cells, step 0.01/0.02 Å |
| `run_c_gfn1.json/.md`, `validation_run_c_gfn1.json/.md` | `runs/20261113-114406_run_c` (Run C, CPU) | xtb-gfn1: CO/Cu(111), CO/Ru(0001) (`validation_*`) and H/Si(100), NH2+H/Si(100), SiOH (`run_c_*`) |
| `run_c_uma.json/.md`, `validation_run_c_uma.json/.md` | `runs/20261113-122442_run_c` (Run C, CUDA) | uma-omol, uma-oc20, uma-omat; NH2+H slab rows from the 2x2 cell with the FIRE retry |

In `run_c_*.md` the comparisons with `data/reference/*.csv` are made when the file is rendered.

Derived files (regenerate, do not edit):
- `adsorbate_sites.json`, `convergence.json`: `scripts/classify_sites.py` over the run directories
  (oldest first). They hold the relaxed site per adsorbate case and the final optimizer fmax per
  sub-run. Not converged: PET-OMol-L CO/Pt (1.22 eV/Å), dxtb CO/Au (0.14), PET-MAD gas CO (0.025).
- `summary_tables.md` and the result figures in `docs/figures/`: `scripts/make_report_figures.py`;
  `docs/figures/agent_flowchart.png` (the workflow diagram): `scripts/make_flowchart.py`.
- `co_pt111_gfn1_cluster_report.md` (+ `docs/figures/co_pt111_gfn1_cluster.png`): an agent-graph run
  of CO on the Pt10 cluster with GFN1-xTB (`xtb --hess` route, fragment dipoles, critic OK).
