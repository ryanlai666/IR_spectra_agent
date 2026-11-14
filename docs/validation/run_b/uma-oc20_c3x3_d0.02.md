# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | uma-oc20 |
|---|---|---|---|
| CO | C-O stretch | 2143 | 2088 (-55) / 27 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3317 (-20) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1022 (+72) / 182 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3446 (+2) / 15 |
| NH3 | deg. deformation | 1627 / 28.2 | 1610 (-17) / 51 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uma-oc20 | GFN2-xTB (xtb) | 5 | 33 | -4 | 72 | 1.004 | 31 | 12.5 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | uma-oc20 | surface (29 at.) | ontop 1.85 | C-O stretch | 2033 | uma-oc20 | -55 | 2068 (unverified) | -75 | 26.2 | - | 4.2 |
| NH3/Pt(111) | uma-oc20 | surface (31 at.) | ontop 2.14 | sym. deformation (umbrella) | 1142 | uma-oc20 | +120 | 1090 (unverified) | +140 | 224.3 | - | 5.4 |
