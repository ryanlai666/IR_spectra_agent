# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | uma-oc20 | uma-omat | uma-omol |
|---|---|---|---|---|---|
| CO | C-O stretch | 2143 | 2086 (-57) / 27 | 2118 (-25) / 27 | 2244 (+101) / 27 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uma-oc20 | GFN2-xTB (xtb) | 1 | 57 | -57 | 57 | 1.027 | 0 | 2.0 |
| uma-omat | GFN2-xTB (xtb) | 1 | 25 | -25 | 25 | 1.012 | 0 | 1.8 |
| uma-omol | GFN2-xTB (xtb) | 1 | 101 | +101 | 101 | 0.955 | 0 | 13.2 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Cu(111) | uma-omol | cluster (12 at.) | ontop 2.10 | C-O stretch | 2154 | uma-omol | -90 | 2077 | -66 | 26.9 | - | 3.3 |
| CO/Ru(0001) | uma-omol | cluster (12 at.) | ontop 1.92 | C-O stretch | 2064 | uma-omol | -180 | 1976 (unverified) | -167 | 26.7 | - | 2.7 |
| CO/Cu(111) | uma-oc20 | surface (29 at.) | ontop 1.87 | C-O stretch | 1970 | uma-oc20 | -116 | 2077 | -66 | 26.3 | - | 5.3 |
| CO/Ru(0001) | uma-oc20 | surface (29 at.) | ontop 1.89 | C-O stretch | 1926 | uma-oc20 | -160 | 1976 (unverified) | -167 | 26.1 | - | 4.5 |
| CO/Cu(111) | uma-omat | surface (29 at.) | ontop 1.84 | C-O stretch | 2015 | uma-omat | -103 | 2077 | -66 | 26.3 | - | 5.2 |
| CO/Ru(0001) | uma-omat | surface (29 at.) | ontop 1.87 | C-O stretch | 1940 | uma-omat | -178 | 1976 (unverified) | -167 | 26.0 | - | 3.5 |
