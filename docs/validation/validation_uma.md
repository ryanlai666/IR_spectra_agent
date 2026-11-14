# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | uma-oc20 | uma-omat | uma-omol |
|---|---|---|---|---|---|
| H2O | OH sym. stretch | 3657 / 2.9 | 3640 (-17) / 5 | 3655 (-2) / 5 | 3819 (+162) / 6 |
| H2O | bend | 1595 / 62.5 | 1585 (-10) / 127 | 1559 (-36) / 129 | 1624 (+29) / 131 |
| H2O | OH asym. stretch | 3756 / 41.7 | 3760 (+4) / 14 | 3790 (+34) / 14 | 3918 (+162) / 15 |
| CO | C-O stretch | 2143 | 2086 (-57) / 27 | 2118 (-25) / 27 | 2244 (+101) / 27 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3316 (-21) / 0 | 3369 (+32) / 0 | 3487 (+150) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1023 (+73) / 182 | 1006 (+56) / 192 | 1025 (+75) / 205 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3445 (+1) / 15 | 3496 (+52) / 13 | 3603 (+159) / 11 |
| NH3 | deg. deformation | 1627 / 28.2 | 1611 (-16) / 51 | 1628 (+1) / 54 | 1660 (+33) / 56 |
| CO2 | sym. stretch | 1333 | 1289 (-44) / 0 | 1315 (-18) / 0 | 1385 (+52) / 0 |
| CO2 | asym. stretch | 2349 / 665 | 2312 (-37) / 957 | 2363 (+14) / 974 | 2405 (+56) / 1011 |
| CO2 | bend | 667 / 54.2 | 625 (-42) / 125 | 603 (-64) / 128 | 684 (+17) / 132 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uma-oc20 | GFN2-xTB (xtb) | 11 | 29 | -15 | 73 | 1.005 | 26 | 7.8 |
| uma-omat | GFN2-xTB (xtb) | 11 | 30 | +4 | 64 | 0.995 | 29 | 8.6 |
| uma-omol | GFN2-xTB (xtb) | 11 | 90 | +90 | 162 | 0.960 | 19 | 16.4 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | uma-omol | cluster (12 at.) | ontop 1.87 | C-O stretch | 2096 | uma-omol | -148 | 2068 (unverified) | -75 | 26.8 | - | 1.9 |
| NH3/Pt(111) | uma-omol | cluster (14 at.) | ontop 2.16 | sym. deformation (umbrella) | 1227 | uma-omol | +202 | 1090 (unverified) | +140 | 218.9 | imaginary_modes | 3.2 |
| CO/Au(111) | uma-omol | cluster (12 at.) | ontop 2.10 | C-O stretch | 2195 | uma-omol | -49 | - (unverified) | - | 27.0 | - | 2.2 |
| NH3/Au(111) | uma-omol | cluster (14 at.) | ontop 2.29 | sym. deformation (umbrella) | 1176 | uma-omol | +152 | - (unverified) | - | 210.9 | - | 3.4 |
| CO/Pt(111) | uma-oc20 | surface (29 at.) | ontop 1.85 | C-O stretch | 2031 | uma-oc20 | -55 | 2068 (unverified) | -75 | 26.2 | - | 4.6 |
| NH3/Pt(111) | uma-oc20 | surface (31 at.) | ontop 2.14 | sym. deformation (umbrella) | 1142 | uma-oc20 | +120 | 1090 (unverified) | +140 | 224.3 | - | 5.2 |
| CO/Au(111) | uma-oc20 | surface (29 at.) | ontop 2.09 | C-O stretch | 2028 | uma-oc20 | -58 | - (unverified) | - | 26.5 | imaginary_modes | 4.7 |
| NH3/Au(111) | uma-oc20 | surface (31 at.) | ontop 2.62 | sym. deformation (umbrella) | 1013 | uma-oc20 | -10 | - (unverified) | - | 206.0 | imaginary_modes | 7.0 |
| CO/Pt(111) | uma-omat | surface (29 at.) | ontop 1.84 | C-O stretch | 2037 | uma-omat | -81 | 2068 (unverified) | -75 | 26.2 | - | 4.2 |
| NH3/Pt(111) | uma-omat | surface (31 at.) | ontop 2.12 | sym. deformation (umbrella) | 1139 | uma-omat | +134 | 1090 (unverified) | +140 | 240.6 | imaginary_modes | 5.1 |
| CO/Au(111) | uma-omat | surface (29 at.) | ontop 2.01 | C-O stretch | 2056 | uma-omat | -62 | - (unverified) | - | 26.5 | - | 4.8 |
| NH3/Au(111) | uma-omat | surface (31 at.) | ontop 2.40 | sym. deformation (umbrella) | 963 | uma-omat | -43 | - (unverified) | - | 244.6 | - | 7.8 |

Critic flags:
- NH3/Pt(111) / uma-omol: imaginary_modes: 1 imaginary mode(s) below -50 cm^-1 (lowest -66 cm^-1)
- CO/Au(111) / uma-oc20: imaginary_modes: 2 imaginary mode(s) below -50 cm^-1 (lowest -57 cm^-1)
- NH3/Au(111) / uma-oc20: imaginary_modes: 1 imaginary mode(s) below -50 cm^-1 (lowest -62 cm^-1)
- NH3/Pt(111) / uma-omat: imaginary_modes: 1 imaginary mode(s) below -50 cm^-1 (lowest -73 cm^-1)
