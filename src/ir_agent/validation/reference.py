"""Reference vibrational data used for validation and in reports.

Gas-phase fundamentals: NIST CCCBDB "experimental vibrational frequencies" pages (checked
2026-11-12), which cite Shimanouchi, *Tables of Molecular Vibrational Frequencies* (NSRDS-NBS
39, 1972) and Huber & Herzberg (1979). These are *anharmonic fundamentals*; harmonic
calculations are expected to overestimate them (X-H stretches by roughly 3-5%).

Experimental *harmonic* frequencies (``HARMONIC``) exist for H2O and CO only among the four test
molecules; they allow a harmonic-vs-harmonic check of the methods without the anharmonicity offset.

Adsorbate values are literature *trends*; see ``verified`` and ``source`` for each entry. Shifts
are compared like with like: experimental fundamental (adsorbed) minus experimental fundamental
(gas) on the reference side, computed harmonic minus computed harmonic of the same method on the
calculation side; ``shift_window`` gives the range a computed harmonic shift should fall in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fundamental:
    mode: str
    symmetry: str
    wavenumber_cm: float
    ir_intensity_km_mol: float | None
    degeneracy: int = 1
    ir_active: bool = True


CCCBDB_URL = "https://cccbdb.nist.gov/exp2x.asp?casno={casno}&charge=0"

GAS_PHASE: dict[str, dict] = {
    "H2O": {
        "casno": "7732185",
        "modes": [
            Fundamental("OH sym. stretch", "A1", 3657, 2.9),
            Fundamental("bend", "A1", 1595, 62.5),
            Fundamental("OH asym. stretch", "B2", 3756, 41.7),
        ],
    },
    "CO": {
        "casno": "630080",
        "modes": [Fundamental("C-O stretch", "Sigma", 2143, None)],
    },
    "NH3": {
        "casno": "7664417",
        "modes": [
            Fundamental("NH sym. stretch", "A1", 3337, 7.1),
            Fundamental("sym. deformation (umbrella)", "A1", 950, 138.0),
            Fundamental("NH deg. stretch", "E", 3444, 4.3, degeneracy=2),
            Fundamental("deg. deformation", "E", 1627, 28.2, degeneracy=2),
        ],
    },
    "CO2": {
        "casno": "124389",
        "modes": [
            Fundamental("sym. stretch", "Sigma_g", 1333, None, ir_active=False),
            Fundamental("asym. stretch", "Sigma_u", 2349, 665.0),
            Fundamental("bend", "Pi_u", 667, 54.2, degeneracy=2),
        ],
    },
}


#: Experimental harmonic wavenumbers (cm^-1), keyed by the mode names of GAS_PHASE.
HARMONIC: dict[str, dict] = {
    "H2O": {
        "modes": {"OH sym. stretch": 3832.2, "bend": 1648.5, "OH asym. stretch": 3942.5},
        "source": "NIST CCCBDB experimental harmonic frequencies, https://cccbdb.nist.gov/exptriatomicsx.asp "
                  "(checked 2026-11-13)",
    },
    "CO": {
        "modes": {"C-O stretch": 2169.8},
        "source": "omega_e of CO X 1Sigma+, Huber & Herzberg (1979) via NIST CCCBDB (checked 2026-11-13)",
    },
}


def harmonic_reference(name: str, mode: str) -> float | None:
    return HARMONIC.get(name, {}).get("modes", {}).get(mode)


@dataclass(frozen=True)
class AdsorbateReference:
    system: str
    band: str
    gas_cm: float
    adsorbed_cm: float | None
    trend: str
    source: str
    verified: bool
    note: str = ""
    #: (low, high) range in which a computed *harmonic* shift counts as consistent (cm^-1)
    shift_window: tuple[float, float] | None = None


ADSORBATE_REFERENCES: dict[tuple[str, str], AdsorbateReference] = {
    ("CO", "Pt"): AdsorbateReference(
        system="CO atop on Pt(111), low coverage (UHV)",
        band="C-O stretch",
        gas_cm=2143,
        adsorbed_cm=2068,
        trend="red shift of -75 to -54 cm^-1 vs the gas fundamental at low coverage; blue-shifts with coverage",
        source=(
            "UHV IRAS, lowest coverage: 2068 cm^-1 (Hayden & Bradshaw, Surf. Sci. 125, 787 (1983), "
            "doi:10.1016/S0039-6028(83)80060-0) and 2089 cm^-1 on a low-defect surface (Tüshaus et al., "
            "J. Electron Spectrosc. Relat. Phenom. 44, 305 (1987), doi:10.1016/0368-2048(87)87031-7); both "
            "seen only as quotes in Molina et al. 2023 (secondary). The often-quoted 2070 cm^-1 comes from "
            "SFG on a Pt(111)/Ru bimetallic electrode, not UHV, and is not used. See "
            "data/reference/experimental_surface_ir.csv."
        ),
        verified=False,
        note=("Window = measured low-coverage shifts (-75 and -54) +/- 30. Harmonic vs fundamental: the "
              "shift of omega_e equals the fundamental shift plus (2 omega_e x_e,ads - 2 omega_e x_e,gas), "
              "0 cm^-1 for CO/Ir(111) and about +10 for CO/Ru(0001) (data/reference/anharmonicity.csv), "
              "well inside the +/-30 tolerance."),
        shift_window=(-105.0, -24.0),
    ),
    ("NH3", "Pt"): AdsorbateReference(
        system="NH3 atop on Pt(111), low coverage",
        band="sym. deformation (umbrella)",
        gas_cm=950,
        adsorbed_cm=1090,
        trend="blue shift of the umbrella mode on adsorption via N (fundamental shift ~+140 cm^-1)",
        source=(
            "EELS 1090 cm^-1 attributed to Gland & Sexton, J. Catal. 68, 286 (1981), "
            "doi:10.1016/0021-9517(81)90098-1, seen only in search-result abstract text. Crossref gives the "
            "title 'Observation of an NH3-NO complex on the Pt(111) surface', so the value may refer to NH3 "
            "co-adsorbed with NO. The primary HREELS study of NH3 alone (Sexton & Mitchell, Surf. Sci. 99, 523 "
            "(1980)) was not accessible; a value near 1170 cm^-1 could not be found again."
        ),
        verified=False,
        note=(
            "Experimental fundamental vs gas fundamental (950 = mean of the inversion doublet 932/968). "
            "The gas umbrella mode is strongly anharmonic (double well): harmonic OMol25 models give "
            "~1025, ~75 above 950, while adsorbed NH3 cannot invert, so a computed harmonic shift is "
            "expected below the +140 fundamental shift. +65 to +220 counts as consistent; the reference "
            "itself is weak (see source)."
        ),
        shift_window=(65.0, 220.0),
    ),
    ("CO", "Cu"): AdsorbateReference(
        system="CO atop on Cu(111), lowest coverage",
        band="C-O stretch",
        gas_cm=2143,
        adsorbed_cm=2077,
        trend="red shift of -66 cm^-1 vs the gas fundamental at the lowest coverage; red-shifts further with coverage",
        source=(
            "RAIRS, clean Cu(111), ~80 K, initial dose: 2077 cm^-1 (Zhang et al., J. Phys. Chem. C 126, 13114 "
            "(2022), doi:10.1021/acs.jpcc.2c02541; read in the full text). 2078 -> 2068 at saturation "
            "(Molina et al. 2023; Hollins & Pritchard 1979)."
        ),
        verified=True,
        note="Window = -66 +/- 30. Saturation (0.52 ML) value 2068 gives -75.",
        shift_window=(-96.0, -36.0),
    ),
    ("CO", "Ru"): AdsorbateReference(
        system="CO atop on Ru(0001), lowest coverage",
        band="C-O stretch",
        gas_cm=2143,
        adsorbed_cm=1976,
        trend="red shift of about -167 to -158 cm^-1 at low coverage; rises to ~2030 at 1/3 ML",
        source=(
            "HREELS, lowest coverage: 245 meV = 1976 cm^-1 (CO-only reference in a Cs+CO/Ru(0001) study, "
            "search snippet, authors not verified); uncoupled 12C16O in the dense layer ~1985 (derived from "
            "Jakob, PRL 77, 4229 (1996), 13C16O at 1941). Pfnür et al., Surf. Sci. 93, 431 (1980) not accessible."
        ),
        verified=False,
        note=("Window = (-167, -158) +/- 30. Adsorbed CO on Ru is more anharmonic than gas CO (omega_e x_e "
              "18.5 vs 13.4, Jakob 1996), so the harmonic shift should be ~+10 above the fundamental shift."),
        shift_window=(-197.0, -128.0),
    ),
    ("CO", "Au"): AdsorbateReference(
        system="CO atop on Au(111)",
        band="C-O stretch",
        gas_cm=2143,
        adsorbed_cm=None,
        trend="small shift vs gas phase: Au binds CO weakly (expected |shift| < Pt case)",
        source=("qualitative trend only: low-temperature IRAS of atop CO on Au(111) exists (Pischel & Pucci, "
                "J. Phys. Chem. C 119, 18340 (2015), doi:10.1021/acs.jpcc.5b05051) but no number was obtained"),
        verified=False,
    ),
    ("NH3", "Au"): AdsorbateReference(
        system="NH3 atop on Au(111)",
        band="sym. deformation (umbrella)",
        gas_cm=950,
        adsorbed_cm=None,
        trend="blue shift of the umbrella mode, smaller than on Pt (weaker N-metal bond)",
        source=("qualitative trend only: no measured umbrella frequency found (Kay et al., J. Chem. Phys. 91, "
                "5120 (1989) is a molecular-beam/TPD study without vibrational data)"),
        verified=False,
    ),
}


def gas_phase_fundamentals(name: str) -> list[Fundamental] | None:
    entry = GAS_PHASE.get(name)
    return None if entry is None else list(entry["modes"])


def expanded_fundamentals(name: str) -> list[Fundamental]:
    """Fundamentals with degenerate modes repeated (3N-5/6 entries)."""
    out: list[Fundamental] = []
    for f in gas_phase_fundamentals(name) or []:
        out.extend([f] * f.degeneracy)
    return out


def cccbdb_url(name: str) -> str | None:
    entry = GAS_PHASE.get(name)
    return None if entry is None else CCCBDB_URL.format(casno=entry["casno"])
