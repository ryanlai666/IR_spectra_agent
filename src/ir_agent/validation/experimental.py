"""Measured surface vibrational bands and anharmonicity data (``data/reference/*.csv``).

``experimental_surface_ir.csv``: one row per measured value with technique, coverage, temperature,
citation, DOI and an evidence level:

* ``primary-fulltext``: the number was read in the primary paper itself (``verified = yes``);
* ``primary-abstract``: only in the primary paper's abstract (``verified = no``);
* ``secondary``: only in a review, a citing paper or a search snippet (``verified = no``).

``compare = yes`` marks the value used for comparisons (lowest coverage where coverage matters).

``anharmonicity.csv``: fundamental (0->1), overtone (0->2) or hot-band (1->2) data for a Morse
oscillator, nu01 = omega_e - 2 omega_e x_e and nu02 = 2 omega_e - 6 omega_e x_e, so

    gap = omega_e - nu01 = 2 omega_e x_e = 2 nu01 - nu02 = nu01 - nu12,   omega_e = 3 nu01 - nu02.

``gap_cm`` is how far a harmonic calculation should lie *above* the measured fundamental (gas CO:
3*2143.3 - 4260.1 = 2169.8 = omega_e). ``applies_to`` lists ``system:band`` keys that use it.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

from ir_agent.config import REPO_ROOT

DATA = REPO_ROOT / "data" / "reference"
EXP_CSV = DATA / "experimental_surface_ir.csv"
ANH_CSV = DATA / "anharmonicity.csv"


@functools.lru_cache(maxsize=4)
def _load(path: str) -> tuple[dict[str, str], ...]:
    p = Path(path)
    if not p.exists():
        return ()
    with open(p, encoding="utf-8", newline="") as fh:
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    return tuple(r for r in rows if r.get("system") or r.get("species"))


def experimental_rows(path: Path = EXP_CSV) -> list[dict[str, str]]:
    return list(_load(str(path)))


def anharmonicity_rows(path: Path = ANH_CSV) -> list[dict[str, str]]:
    return list(_load(str(path)))


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def comparison_value(system: str, band: str, path: Path = EXP_CSV) -> dict | None:
    """The measured value used for ``system``/``band`` (row with ``compare = yes``), or None."""
    for r in _load(str(path)):
        if r.get("system") == system and r.get("band") == band and r.get("compare", "").lower() == "yes":
            out = dict(r)
            out["value"] = _num(r.get("value_cm", ""))
            out["verified_bool"] = r.get("verified", "").lower() == "yes"
            return out
    return None


def anharmonic_gap(system: str, band: str, path: Path = ANH_CSV) -> dict | None:
    """Harmonic-minus-fundamental gap that applies to ``system``/``band`` (from an overtone), or None."""
    key = f"{system}:{band}"
    for r in _load(str(path)):
        keys = [k.strip() for k in r.get("applies_to", "").split(";") if k.strip()]
        if key in keys:
            out = dict(r)
            out["gap"] = _num(r.get("gap_cm", ""))
            out["verified_bool"] = r.get("verified", "").lower() == "yes"
            return out
    return None
