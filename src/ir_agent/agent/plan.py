"""Simulation plan schema shared by the LLM planner and the deterministic --no-llm mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ir_agent.calculators.registry import canonical_preset
from ir_agent.structures import SLAB_BUILDERS, is_element, validate_molecule_name

_NONE_STRINGS = {"", "none", "null", "no", "gas", "gas-phase", "vacuum", "n/a"}


class SurfaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metal: str
    structure: Literal["fcc", "bcc", "hcp"] = "fcc"
    facet: str = "111"
    site: str = "ontop"
    #: slab = periodic slab; cluster = finite metal cluster (non-periodic methods);
    #: auto = slab if the calculator is periodic, otherwise cluster
    model: Literal["auto", "slab", "cluster"] = "auto"

    @field_validator("metal")
    @classmethod
    def _metal(cls, v: str) -> str:
        v = v.strip()
        v = v[:1].upper() + v[1:].lower()
        if not is_element(v):
            raise ValueError(f"'{v}' is not an element symbol (e.g. Au, Pt, Cu)")
        return v

    @field_validator("facet")
    @classmethod
    def _facet(cls, v: str) -> str:
        return v.strip().strip("()").replace("-1", "m1")

    @model_validator(mode="after")
    def _facet_for_structure(self) -> SurfaceSpec:
        allowed = SLAB_BUILDERS[self.structure]
        if self.facet not in allowed:
            raise ValueError(
                f"facet '{self.facet}' not supported for {self.structure}; choose from {sorted(allowed)}"
            )
        return self


class SimulationPlan(BaseModel):
    """Validated, normalized plan. Serializable as the --no-llm JSON plan."""

    model_config = ConfigDict(extra="forbid")

    adsorbate: str = Field(description="ASE g2 molecule name, e.g. H2O, CO, NH3, CO2")
    calculator: str = Field(default="emt", description="calculator preset (see `ir-agent calculators`)")
    surface: SurfaceSpec | None = None
    dipole_method: Literal["auto", "native", "xtb", "xtb-gfn1", "tblite", "dxtb", "none"] = "auto"
    charge: int = 0
    spin_multiplicity: int | None = Field(default=None, ge=1)
    task: str | None = Field(default=None, description="original natural-language request")

    @field_validator("adsorbate")
    @classmethod
    def _adsorbate(cls, v: str) -> str:
        return validate_molecule_name(v.strip())

    @field_validator("calculator")
    @classmethod
    def _calculator(cls, v: str) -> str:
        return canonical_preset(v)

    @property
    def is_surface(self) -> bool:
        return self.surface is not None

    @property
    def label(self) -> str:
        if self.surface is None:
            return f"{self.adsorbate}_gas_{self.calculator}"
        s = self.surface
        return f"{self.adsorbate}_on_{s.metal}{s.facet}_{s.structure}_{self.calculator}"

    @property
    def description(self) -> str:
        if self.surface is None:
            return f"{self.adsorbate} (gas phase) with {self.calculator}"
        s = self.surface
        return f"{self.adsorbate} on {s.metal}({s.facet}) [{s.structure}, {s.site}] with {self.calculator}"


class PlannerOutput(BaseModel):
    """Flat schema for LLM structured output (simpler for function calling)."""

    calculator: str = Field(
        description=(
            "Calculator preset: 'emt' (fast toy), 'uma' (Meta UMA ML potential, GPU), 'uma-omol', "
            "'uma-oc20', 'uma-omat' (UMA with one task), 'orb-mol', 'orb-omat', 'pet-omol', 'pet-mad', "
            "'xtb' (GFN2-xTB), 'xtb-gfn1', 'tblite', 'dxtb' (GFN1-xTB), 'dftb+', 'gpaw', 'nwchem', "
            "'vasp', 'vasp-accurate'. Default 'emt' when unspecified."
        )
    )
    adsorbate: str = Field(description="Molecule as an ASE g2 name, e.g. 'H2O', 'CO', 'NH3', 'CO2'.")
    surface_metal: str = Field(
        description="Element symbol of the metal surface (e.g. 'Au', 'Pt'), or 'None' for an isolated gas-phase molecule."
    )
    surface_structure: str = Field(default="fcc", description="'fcc', 'bcc' or 'hcp'.")
    surface_facet: str = Field(default="111", description="Miller index, e.g. '111', '100', '110', '0001'.")
    dipole_method: str = Field(default="auto", description="'auto' unless the user names a dipole method.")
    surface_site: str = Field(default="ontop", description="'ontop', 'bridge', 'fcc' or 'hcp'.")
    surface_model: str = Field(default="auto", description="'auto', 'slab' or 'cluster'.")

    def to_plan(self, task: str | None = None) -> SimulationPlan:
        surface = None
        if self.surface_metal.strip().lower() not in _NONE_STRINGS:
            site = (self.surface_site or "ontop").strip().lower()
            site = {"hollow": "fcc", "atop": "ontop", "top": "ontop", "on-top": "ontop"}.get(site, site)
            surface = SurfaceSpec(
                metal=self.surface_metal,
                structure=self.surface_structure.strip().lower() or "fcc",  # type: ignore[arg-type]
                facet=self.surface_facet,
                site=site,
                model=(self.surface_model or "auto").strip().lower(),  # type: ignore[arg-type]
            )
        return SimulationPlan(
            adsorbate=self.adsorbate,
            calculator=self.calculator,
            surface=surface,
            dipole_method=(self.dipole_method or "auto").lower(),  # type: ignore[arg-type]
            task=task,
        )


# v0.1 plan keys -> v0.2
_LEGACY_KEYS = {
    "calculator_level": "calculator",
    "adsorbate_formula": "adsorbate",
    "surface_metal_structure_type": "surface_structure",
    "surface_metal_facet": "surface_facet",
}


def plan_from_dict(data: dict[str, Any]) -> SimulationPlan:
    """Accept the nested v0.2 schema, the flat LLM schema, or v0.1 planner keys."""
    if "surface_metal" in data or any(k in data for k in _LEGACY_KEYS):
        flat = {_LEGACY_KEYS.get(k, k): v for k, v in data.items()}
        task = flat.pop("task", None)
        flat.setdefault("calculator", "emt")
        if flat.get("surface_metal") is None:
            flat["surface_metal"] = "None"
        extra = {k: flat.pop(k) for k in ("charge", "spin_multiplicity") if k in flat}
        plan = PlannerOutput(**flat).to_plan(task)
        return SimulationPlan.model_validate({**plan.model_dump(), **extra}) if extra else plan
    return SimulationPlan.model_validate(data)


def load_plan(source: str | Path | dict[str, Any]) -> SimulationPlan:
    if isinstance(source, dict):
        return plan_from_dict(source)
    text = Path(source).read_text(encoding="utf-8")
    return plan_from_dict(json.loads(text))
