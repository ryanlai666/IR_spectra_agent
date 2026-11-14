# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | pet-mad | pet-omol |
|---|---|---|---|---|
| H2O | OH sym. stretch | 3657 / 2.9 | 3765 (+108) / 6 | 3823 (+166) / 6 |
| H2O | bend | 1595 / 62.5 | 1579 (-16) / 132 | 1621 (+26) / 131 |
| H2O | OH asym. stretch | 3756 / 41.7 | 3830 (+74) / 14 | 3921 (+165) / 15 |
| CO | C-O stretch | 2143 | 2208 (+65) / 27 | 2239 (+96) / 27 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3502 (+165) / 0 | 3493 (+156) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1046 (+96) / 199 | 1025 (+75) / 205 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3564 (+120) / 12 | 3608 (+164) / 11 |
| NH3 | deg. deformation | 1627 / 28.2 | 1606 (-21) / 56 | 1654 (+27) / 56 |
| CO2 | sym. stretch | 1333 | 1313 (-20) / 0 | 1388 (+55) / 0 |
| CO2 | asym. stretch | 2349 / 665 | 2299 (-50) / 1001 | 2414 (+65) / 1010 |
| CO2 | bend | 667 / 54.2 | 563 (-104) / 131 | 684 (+17) / 132 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pet-mad | GFN2-xTB (xtb) | 11 | 76 | +38 | 165 | 0.977 | 57 | 35.2 |
| pet-omol | GFN2-xTB (xtb) | 11 | 92 | +92 | 166 | 0.959 | 19 | 114.8 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).
- CO/pet-mad: flagged not_converged

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | pet-omol | cluster (12 at.) | bridge 1.98/2.00 | C-O stretch | 1795 | pet-omol | -444 | 2068 (unverified) | -75 | 24.2 | not_converged, site_changed | 56.1 |
| NH3/Pt(111) | pet-omol | cluster (14 at.) | ontop 2.43 | sym. deformation (umbrella) | 1032 | pet-omol | +7 | 1090 (unverified) | +140 | 235.1 | - | 50.0 |
| CO/Au(111) | pet-omol | cluster (12 at.) | ontop 2.21 | C-O stretch | 2111 | pet-omol | -127 | - (unverified) | - | 27.0 | - | 27.4 |
| NH3/Au(111) | pet-omol | cluster (14 at.) | ontop 2.42 | sym. deformation (umbrella) | 1014 | pet-omol | -11 | - (unverified) | - | 233.0 | - | 35.4 |
| CO/Pt(111) | pet-mad | surface (29 at.) | ontop 1.85 | C-O stretch | 2091 | pet-mad | -118 | 2068 (unverified) | -75 | 26.7 | - | 13.2 |
| NH3/Pt(111) | pet-mad | surface (31 at.) | ontop 2.11 | sym. deformation (umbrella) | 1141 | pet-mad | +95 | 1090 (unverified) | +140 | 241.3 | - | 23.3 |
| CO/Au(111) | pet-mad | surface (29 at.) | ontop 1.99 | C-O stretch | 2138 | pet-mad | -71 | - (unverified) | - | 26.8 | - | 16.1 |
| NH3/Au(111) | pet-mad | surface (31 at.) | ontop 2.39 | sym. deformation (umbrella) | 1051 | pet-mad | +5 | - (unverified) | - | 238.5 | - | 24.0 |
