# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | uma-omat |
|---|---|---|---|
| CO | C-O stretch | 2143 | 2118 (-25) / 27 |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3369 (+32) / 0 |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1006 (+56) / 192 |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3496 (+52) / 13 |
| NH3 | deg. deformation | 1627 / 28.2 | 1628 (+1) / 54 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uma-omat | GFN2-xTB (xtb) | 5 | 33 | +23 | 56 | 0.991 | 25 | 12.2 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | uma-omat | surface (50 at.) | ontop 1.84 | C-O stretch | 2031 | uma-omat | -88 | 2068 (unverified) | -75 | 26.2 | - | 4.2 |
| CO/Pt(111)/bridge | uma-omat | surface (50 at.) | bridge 2.01/2.01 | C-O stretch | 1832 | uma-omat | -286 | 2068 (unverified) | -75 | 25.8 | - | 7.4 |
| CO/Pt(111)/fcc | uma-omat | surface (50 at.) | fcc 2.11/2.11/2.11 | C-O stretch | 1747 | uma-omat | -371 | 2068 (unverified) | -75 | 25.7 | - | 4.9 |
| NH3/Pt(111) | uma-omat | surface (52 at.) | ontop 2.12 | sym. deformation (umbrella) | 1141 | uma-omat | +136 | 1090 (unverified) | +140 | 240.0 | imaginary_modes | 5.4 |

Critic flags:
- NH3/Pt(111) / uma-omat: imaginary_modes: 1 imaginary mode(s) below -50 cm^-1 (lowest -84 cm^-1)
