# Run C (uma-omol, uma-oc20, uma-omat)

## CO on Cu(111) and Ru(0001)

See `validation_run_c_uma.md` in this directory.

## Si / SiO2 systems: absolute wavenumbers vs experiment (cm^-1)

calc = harmonic (partial Hessian: adsorbate + top layer). exp = measured fundamental (data/reference/experimental_surface_ir.csv; '(unverified)' = not read in the primary paper). gap = harmonic minus fundamental from overtone data (anharmonicity.csv); harm. est. = exp + gap.

| system | calculator | model | band | calc | exp | calc - exp | gap | harm. est. | calc - harm. est. | I (km/mol) | flags |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| H/Si(100)-2x1 | uma-omol | cluster (H14Si9) | Si-H stretch (sym) | 2235 | 2099 (unverified) | +136 | +90 | 2189 | +46 | 577 | - |
| H/Si(100)-2x1 | uma-omol | cluster (H14Si9) | Si-H stretch (asym) | 2227 | 2088 (unverified) | +139 | +90 | 2178 | +49 | 94 | - |
| NH2+H/Si(100)-2x1 | uma-omol | cluster (H15NSi9) | Si-H stretch | 2212 | 2077 (unverified) | +135 | +90 | 2167 | +45 | 365 | - |
| NH2+H/Si(100)-2x1 | uma-omol | cluster (H15NSi9) | N-H stretch (asym) | 3658 | - | - | - | - | - | 0 | - |
| NH2+H/Si(100)-2x1 | uma-omol | cluster (H15NSi9) | N-H stretch (sym) | 3569 | - | - | - | - | - | 4 | - |
| NH2+H/Si(100)-2x1 | uma-omol | cluster (H15NSi9) | NH2 scissor | 1566 | 1534 (unverified) | +32 | - | - | - | 53 | - |
| SiOH/SiO2 | uma-omol | cluster (H10O4Si4) | O-H stretch | 3896 | 3747 | +149 | +178 | 3925 | -29 | 15 | - |
| H/Si(100)-2x1 | uma-oc20 | slab (H6Si12) | Si-H stretch (sym) | 2092 | 2099 (unverified) | -7 | +90 | 2189 | -97 | n/a | - |
| H/Si(100)-2x1 | uma-oc20 | slab (H6Si12) | Si-H stretch (asym) | 2081 | 2088 (unverified) | -7 | +90 | 2178 | -97 | n/a | - |
| NH2+H/Si(100)-2x1 | uma-oc20 | slab (H13NSi24) | Si-H stretch | 2067 | 2077 (unverified) | -10 | +90 | 2167 | -100 | n/a | - |
| NH2+H/Si(100)-2x1 | uma-oc20 | slab (H13NSi24) | N-H stretch (asym) | 3496 | - | - | - | - | - | n/a | - |
| NH2+H/Si(100)-2x1 | uma-oc20 | slab (H13NSi24) | N-H stretch (sym) | 3401 | - | - | - | - | - | n/a | - |
| NH2+H/Si(100)-2x1 | uma-oc20 | slab (H13NSi24) | NH2 scissor | 1512 | 1534 (unverified) | -22 | - | - | - | n/a | - |
| SiOH/SiO2 | uma-oc20 | slab (H10O4Si4) | O-H stretch | 3747 | 3747 | -0 | +178 | 3925 | -178 | 15 | - |
| H/Si(100)-2x1 | uma-omat | slab (H6Si12) | Si-H stretch (sym) | 2106 | 2099 (unverified) | +7 | +90 | 2189 | -83 | n/a | - |
| H/Si(100)-2x1 | uma-omat | slab (H6Si12) | Si-H stretch (asym) | 2094 | 2088 (unverified) | +6 | +90 | 2178 | -84 | n/a | - |
| NH2+H/Si(100)-2x1 | uma-omat | slab (H13NSi24) | Si-H stretch | 2089 | 2077 (unverified) | +12 | +90 | 2167 | -78 | n/a | - |
| NH2+H/Si(100)-2x1 | uma-omat | slab (H13NSi24) | N-H stretch (asym) | 3541 | - | - | - | - | - | n/a | - |
| NH2+H/Si(100)-2x1 | uma-omat | slab (H13NSi24) | N-H stretch (sym) | 3444 | - | - | - | - | - | n/a | - |
| NH2+H/Si(100)-2x1 | uma-omat | slab (H13NSi24) | NH2 scissor | 1508 | 1534 (unverified) | -26 | - | - | - | n/a | - |
| SiOH/SiO2 | uma-omat | slab (H10O4Si4) | O-H stretch | 3838 | 3747 | +91 | +178 | 3925 | -87 | 16 | - |
