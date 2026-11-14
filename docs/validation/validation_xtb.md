# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | xtb | xtb-gfn1 |
|---|---|---|---|---|
| H2O | OH sym. stretch | 3657 / 2.9 | 3642 (-15) / 7 | 3676 (+19) / 47 |
| H2O | bend | 1595 / 62.5 | 1540 (-55) / 133 | 1438 (-157) / 240 |
| H2O | OH asym. stretch | 3756 / 41.7 | 3651 (-105) / 17 | 3788 (+32) / 51 |
| CO | C-O stretch | 2143 | 2222 (+79) / 27 | 2241 (+98) / 108 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3426 (+89) / 0 | 3463 (+126) / 9 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1160 (+210) / 211 | 1050 (+100) / 377 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3478 (+34) / 11 | 3571 (+127) / 1 |
| NH3 | deg. deformation | 1627 / 28.2 | 1593 (-34) / 55 | 1611 (-16) / 126 |
| CO2 | sym. stretch | 1333 | 1425 (+92) / 0 | 1446 (+113) / 0 |
| CO2 | asym. stretch | 2349 / 665 | 2593 (+244) / 1047 | 2611 (+262) / 1128 |
| CO2 | bend | 667 / 54.2 | 600 (-67) / 137 | 628 (-39) / 270 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xtb | GFN2-xTB (native) | 11 | 93 | +43 | 244 | 0.986 | 92 | 28.9 |
| xtb-gfn1 | GFN1-xTB (native) | 11 | 99 | +61 | 262 | 0.973 | 82 | 34.1 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | xtb | cluster (12 at.) | ontop 1.82 | C-O stretch | 2059 | xtb | -164 | 2068 (unverified) | -75 | 26.3 | - | 28.5 |
| NH3/Pt(111) | xtb | cluster (14 at.) | ontop 2.47 | sym. deformation (umbrella) | 936 | xtb | -224 | 1090 (unverified) | +140 | 273.1 | - | 49.5 |
| CO/Au(111) | xtb | - | - | - | err: CalculatorError: xtb failed (exit 1). SCC did not converge (seen for metallic Au | | | | | | | 3.2 |
| NH3/Au(111) | xtb | - | - | - | err: CalculatorError: xtb failed (exit 1). SCC did not converge (seen for metallic Au | | | | | | | 3.0 |
| CO/Pt(111) | xtb-gfn1 | cluster (12 at.) | ontop 1.89 | C-O stretch | 2163 | xtb-gfn1 | -78 | 2068 (unverified) | -75 | 26.9 | - | 25.4 |
| NH3/Pt(111) | xtb-gfn1 | cluster (14 at.) | ontop 2.01 | sym. deformation (umbrella) | 1241 | xtb-gfn1 | +192 | 1090 (unverified) | +140 | 262.2 | - | 42.1 |
| CO/Au(111) | xtb-gfn1 | cluster (12 at.) | physisorbed 2.98 | C-O stretch | 2271 | xtb-gfn1 | +30 | - (unverified) | - | 27.0 | imaginary_modes, weakly_bound | 22.6 |
| NH3/Au(111) | xtb-gfn1 | cluster (14 at.) | ontop 2.67 | sym. deformation (umbrella) | 1124 | xtb-gfn1 | +74 | - (unverified) | - | 227.6 | - | 40.2 |

Critic flags:
- CO/Au(111) / xtb-gfn1: imaginary_modes: 2 imaginary mode(s) below -50 cm^-1 (lowest -59 cm^-1)
- CO/Au(111) / xtb-gfn1: weakly_bound: adsorbate physisorbed: shortest C-metal contact 2.98 A > bonding cutoff 2.76 A

Whole-system dipole intensities (native xtb; include metal polarization of the finite cluster, not comparable with the fragment column):
- CO/Pt(111) / xtb: whole-cluster dipole (GFN2-xTB): 1010 km/mol
- NH3/Pt(111) / xtb: whole-cluster dipole (GFN2-xTB): 439 km/mol
- CO/Pt(111) / xtb-gfn1: whole-cluster dipole (GFN1-xTB): 1534 km/mol
- NH3/Pt(111) / xtb-gfn1: whole-cluster dipole (GFN1-xTB): 231 km/mol
- CO/Au(111) / xtb-gfn1: whole-cluster dipole (GFN1-xTB): 175 km/mol
- NH3/Au(111) / xtb-gfn1: whole-cluster dipole (GFN1-xTB): 372 km/mol
