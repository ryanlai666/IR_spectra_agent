# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | uma-omat |
|---|---|---|---|
| CO | C-O stretch | 2143 | 2121 (-22) / 27 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3370 (+33) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1004 (+54) / 192 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3498 (+54) / 13 |
| NH3 | deg. deformation | 1627 / 28.2 | 1628 (+1) / 54 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uma-omat | GFN2-xTB (xtb) | 5 | 33 | +24 | 54 | 0.990 | 24 | 12.9 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | uma-omat | surface (29 at.) | ontop 1.84 | C-O stretch | 2039 | uma-omat | -82 | 2068 (unverified) | -75 | 26.2 | - | 4.1 |
| NH3/Pt(111) | uma-omat | surface (31 at.) | ontop 2.12 | sym. deformation (umbrella) | 1138 | uma-omat | +134 | 1090 (unverified) | +140 | 240.6 | imaginary_modes | 5.2 |

Critic flags:
- NH3/Pt(111) / uma-omat: imaginary_modes: 1 imaginary mode(s) below -50 cm^-1 (lowest -86 cm^-1)
