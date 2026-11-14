# Run C (xtb-gfn1)

## CO on Cu(111) and Ru(0001)

See `validation_run_c_gfn1.md` in this directory.

## Si / SiO2 systems: absolute wavenumbers vs experiment (cm^-1)

calc = harmonic (partial Hessian: adsorbate + top layer). exp = measured fundamental (data/reference/experimental_surface_ir.csv; '(unverified)' = not read in the primary paper). gap = harmonic minus fundamental from overtone data (anharmonicity.csv); harm. est. = exp + gap.

| system | calculator | model | band | calc | exp | calc - exp | gap | harm. est. | calc - harm. est. | I (km/mol) | flags |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| H/Si(100)-2x1 | xtb-gfn1 | cluster (H14Si9) | Si-H stretch (sym) | 2040 | 2099 (unverified) | -59 | +90 | 2189 | -149 | 591 | - |
| H/Si(100)-2x1 | xtb-gfn1 | cluster (H14Si9) | Si-H stretch (asym) | 2038 | 2088 (unverified) | -50 | +90 | 2178 | -140 | 102 | - |
| NH2+H/Si(100)-2x1 | xtb-gfn1 | cluster (H15NSi9) | Si-H stretch | 2032 | 2077 (unverified) | -45 | +90 | 2167 | -135 | 372 | - |
| NH2+H/Si(100)-2x1 | xtb-gfn1 | cluster (H15NSi9) | N-H stretch (asym) | 3502 | - | - | - | - | - | 0 | - |
| NH2+H/Si(100)-2x1 | xtb-gfn1 | cluster (H15NSi9) | N-H stretch (sym) | 3451 | - | - | - | - | - | 4 | - |
| NH2+H/Si(100)-2x1 | xtb-gfn1 | cluster (H15NSi9) | NH2 scissor | 1507 | 1534 (unverified) | -27 | - | - | - | 61 | - |
| SiOH/SiO2 | xtb-gfn1 | cluster (H10O4Si4) | O-H stretch | 3883 | 3747 | +136 | +178 | 3925 | -42 | 79 | - |
