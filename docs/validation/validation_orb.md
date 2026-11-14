# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | orb-mol | orb-omat |
|---|---|---|---|---|
| H2O | OH sym. stretch | 3657 / 2.9 | 3823 (+166) / 6 | 3735 (+78) / 4 |
| H2O | bend | 1595 / 62.5 | 1621 (+26) / 131 | 1548 (-47) / 128 |
| H2O | OH asym. stretch | 3756 / 41.7 | 3923 (+167) / 15 | 3863 (+107) / 14 |
| CO | C-O stretch | 2143 | 2239 (+96) / 27 | 2106 (-37) / 26 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3488 (+151) / 0 | 3404 (+67) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1025 (+75) / 205 | 900 (-50) / 206 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3605 (+161) / 11 | 3547 (+103) / 12 |
| NH3 | deg. deformation | 1627 / 28.2 | 1658 (+31) / 56 | 1604 (-23) / 52 |
| CO2 | sym. stretch | 1333 | 1390 (+57) / 0 | 1338 (+5) / 0 |
| CO2 | asym. stretch | 2349 / 665 | 2419 (+70) / 1010 | 2437 (+88) / 967 |
| CO2 | bend | 667 / 54.2 | 682 (+15) / 132 | 600 (-67) / 127 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| orb-mol | GFN2-xTB (xtb) | 11 | 92 | +92 | 167 | 0.959 | 18 | 64.7 |
| orb-omat | GFN2-xTB (xtb) | 11 | 61 | +20 | 107 | 0.983 | 46 | 16.6 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | orb-mol | cluster (12 at.) | ontop 2.05 | C-O stretch | 2195 | orb-mol | -44 | 2068 (unverified) | -75 | 26.9 | - | 17.2 |
| NH3/Pt(111) | orb-mol | cluster (14 at.) | ontop 2.14 | sym. deformation (umbrella) | 1185 | orb-mol | +160 | 1090 (unverified) | +140 | 238.7 | - | 29.0 |
| CO/Au(111) | orb-mol | cluster (12 at.) | ontop 2.29 | C-O stretch | 2148 | orb-mol | -91 | - (unverified) | - | 26.9 | - | 11.0 |
| NH3/Au(111) | orb-mol | cluster (14 at.) | ontop 2.37 | sym. deformation (umbrella) | 1189 | orb-mol | +164 | - (unverified) | - | 201.6 | - | 25.4 |
| CO/Pt(111) | orb-omat | surface (29 at.) | ontop 1.84 | C-O stretch | 2071 | orb-omat | -35 | 2068 (unverified) | -75 | 26.3 | - | 29.9 |
| NH3/Pt(111) | orb-omat | surface (31 at.) | ontop 2.08 | sym. deformation (umbrella) | 1149 | orb-omat | +249 | 1090 (unverified) | +140 | 234.6 | - | 43.1 |
| CO/Au(111) | orb-omat | surface (29 at.) | ontop 1.98 | C-O stretch | 2058 | orb-omat | -48 | - (unverified) | - | 26.5 | - | 34.1 |
| NH3/Au(111) | orb-omat | surface (31 at.) | ontop 2.50 | sym. deformation (umbrella) | 962 | orb-omat | +62 | - (unverified) | - | 228.9 | - | 44.7 |
