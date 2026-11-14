# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | dxtb |
|---|---|---|---|
| H2O | OH sym. stretch | 3657 / 2.9 | 3676 (+19) |
| H2O | bend | 1595 / 62.5 | 1437 (-158) |
| H2O | OH asym. stretch | 3756 / 41.7 | 3788 (+32) |
| CO | C-O stretch | 2143 | 2242 (+99) |
| NH3 | NH sym. stretch | 3337 / 7.1 | 3464 (+127) |
| NH3 | sym. deformation (umbrella) | 950 / 138 | 1049 (+99) |
| NH3 | NH deg. stretch | 3444 / 4.3 | 3572 (+128) |
| NH3 | deg. deformation | 1627 / 28.2 | 1611 (-16) |
| CO2 | sym. stretch | 1333 | 1447 (+114) |
| CO2 | asym. stretch | 2349 / 665 | 2612 (+263) |
| CO2 | bend | 667 / 54.2 | 628 (-39) |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dxtb | dxtb (native) [Mulliken-charge dipole] | 11 | 99 | +61 | 263 | 0.973 | 82 | 44.6 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Pt(111) | dxtb | - | - | - | err: CalculatorError: dxtb SCF diverged (max |F| = 1.46e+27 eV/A). The pure-PyTorch d | | | | | | | 5.2 |
| NH3/Pt(111) | dxtb | - | - | - | err: CalculatorError: dxtb SCF diverged (max |F| = 4.41e+24 eV/A). The pure-PyTorch d | | | | | | | 5.6 |
| CO/Au(111) | dxtb | cluster (12 at.) | physisorbed 3.00 | C-O stretch | 2270 | dxtb | +29 | - (unverified) | - | - | not_converged, weakly_bound | 84.2 |
| NH3/Au(111) | dxtb | cluster (14 at.) | ontop 2.68 | sym. deformation (umbrella) | 1124 | dxtb | +75 | - (unverified) | - | - | - | 57.6 |
