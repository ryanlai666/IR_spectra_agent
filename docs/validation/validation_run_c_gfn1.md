# Validation

## Gas phase vs NIST CCCBDB experimental fundamentals (cm^-1)

Harmonic frequencies vs anharmonic fundamentals; error = calc - exp. Intensities in km/mol (calc / exp).

| molecule | mode | exp | xtb-gfn1 |
|---|---|---|---|
| CO | C-O stretch | 2143 | 2241 (+98) / 108 |

### Error summary per calculator

| calculator | dipoles | n modes | MAE | mean signed | max abs | LSQ scale* | MAE scaled* | wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xtb-gfn1 | GFN1-xTB (native) | 1 | 98 | +98 | 98 | 0.956 | 0 | 2.7 |

*Scale factor fitted in-sample on these 4 molecules (optimistic; not a transferable factor).

## Adsorbates vs literature trends

Shift = adsorbed band minus the gas band of the same method (and, for UMA, the same task: column 'gas ref'). Rows with critic flags are listed below the table and are not used for conclusions.

| system | calculator | model | relaxed site | band | calc (cm^-1) | gas ref | calc shift | ref adsorbed | ref shift | I (km/mol, fragment, MSSR) | flags | wall (s) |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|
| CO/Cu(111) | xtb-gfn1 | cluster (12 at.) | ontop 2.50 | C-O stretch | 2234 | xtb-gfn1 | -7 | 2077 | -66 | 27.0 | imaginary_modes | 11.4 |
| CO/Ru(0001) | xtb-gfn1 | cluster (12 at.) | ontop 1.95 | C-O stretch | 2093 | xtb-gfn1 | -148 | 1976 (unverified) | -167 | 26.8 | - | 15.8 |

Critic flags:
- CO/Cu(111) / xtb-gfn1: imaginary_modes: 2 imaginary mode(s) below -50 cm^-1 (lowest -69 cm^-1)

Whole-system dipole intensities (native xtb; include metal polarization of the finite cluster, not comparable with the fragment column):
- CO/Cu(111) / xtb-gfn1: whole-cluster dipole (GFN1-xTB): 450 km/mol
- CO/Ru(0001) / xtb-gfn1: whole-cluster dipole (GFN1-xTB): 1665 km/mol
