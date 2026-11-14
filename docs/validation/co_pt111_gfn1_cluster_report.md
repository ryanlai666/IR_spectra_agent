# IR report: CO on Pt(111) [fcc, ontop] with xtb-gfn1

- Force/Hessian method: `GFN1-xTB (xtb --hess)`
- Dipole method: `GFN2-xTB (xtb) on adsorbate fragment`
- Route: `xtb-hessian`; displaced geometries: 12
- Metal-surface selection rule: on (only the surface-normal dipole derivative)
- Timings (s): dipoles=1.40, xtb_hessian=2.50, final_opt=0.25
- Note: Metal-surface selection rule applied: intensities use only the surface-normal component of dmu/dQ.
- Note: Reported intensities use the adsorbate-fragment dipole (same model as the MLIP rows); the whole-cluster GFN1-xTB dipole (includes metal polarization of a finite cluster) is stored separately and is not comparable.

## Modes

| # | wavenumber (cm^-1) | intensity (km/mol) |
|---|---:|---:|
| 0 | 80.4 | 0.00 |
| 1 | 80.4 | 0.00 |
| 2 | 354.1 | 0.00 |
| 3 | 354.1 | 0.00 |
| 4 | 425.7 | 0.04 |
| 5 | 2162.7 | 26.87 |

## Literature trend: CO atop on Pt(111)

- Band: C-O stretch; gas-phase fundamental 2143 cm^-1; adsorbed 2070 cm^-1
- Expected trend: red shift (~-40 to -75 cm^-1 vs gas fundamental); strong, perpendicular dipole
- Source: SFG value ~2070 cm^-1 (Lu, White & Wieckowski, Surf. Sci. 564, 131 (2004)) as quoted by Dabo, Wieckowski & Marzari, arXiv:0711.0174
- A computed harmonic shift (vs the same method's gas value) between -100 and -40 cm^-1 counts as consistent. Value rises with coverage (EELS ~2110 cm^-1 near saturation); low-coverage limit is lower. CO anharmonicity (2*omega_e*x_e ~ 27 cm^-1) is similar in gas and adsorbed CO, so the harmonic shift is close to the fundamental shift (-73).
- Computed C-O stretch: 2163 cm^-1 (mode 5)

## Critic checks

- Attempt 1: OK
- Relaxed site: ontop 1.89 (requested ontop)
