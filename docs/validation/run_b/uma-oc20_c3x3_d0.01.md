# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | uma-oc20 |
|---|---|---|---|
| CO | C-O stretch | 2143 | 2086 (-57) / 27 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3316 (-21) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1023 (+73) / 182 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3445 (+1) / 15 |
| NH3 | deg. deformation | 1627 / 28.2 | 1611 (-16) / 51 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uma-oc20 | GFN2-xTB (xtb) | 5 | 34 | -4 | 73 | 1.004 | 31 | 13.7 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | uma-oc20 | surface (29 at.) | ontop 1.85 | C-O stretch | 2031 | uma-oc20 | -55 | 2068 (unverified) | -75 | 26.2 | - | 5.5 |
| CO/Pt(111)/bridge | uma-oc20 | surface (29 at.) | bridge 2.02/2.02 | C-O stretch | 1829 | uma-oc20 | -256 | 2068 (unverified) | -75 | 25.8 | - | 4.9 |
| CO/Pt(111)/fcc | uma-oc20 | surface (29 at.) | fcc 2.12/2.12/2.12 | C-O stretch | 1740 | uma-oc20 | -346 | 2068 (unverified) | -75 | 25.6 | - | 5.1 |
| NH3/Pt(111) | uma-oc20 | surface (31 at.) | ontop 2.14 | sym. deformation (umbrella) | 1142 | uma-oc20 | +120 | 1090 (unverified) | +140 | 224.3 | - | 6.0 |
