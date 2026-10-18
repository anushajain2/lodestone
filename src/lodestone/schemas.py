"""Data contracts: the standard experiment record, import reports and campaign specs (D2, T1).

Every lab format is converted into :class:`ExperimentRecord`. Units are explicit and canonical
(converted with ``pint`` by the adapters), and every record carries provenance back to the source
sample, library and measurement type, so any number on screen can be traced to where it came from.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm "
    "Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu".split()
)

PropertyName = Literal["conductivity", "band_gap", "visible_transmittance", "sheet_resistance", "thickness"]

# Canonical units for each property, and how the property is modelled (log for conductivity).
PROPERTIES: dict[str, dict] = {
    "conductivity": {"unit": "S/cm", "label": "Conductivity", "log": True, "method": "four-point probe"},
    "band_gap": {"unit": "eV", "label": "Direct band gap", "log": False, "method": "UV-vis-NIR transmittance/reflectance (Tauc, direct)"},
    "visible_transmittance": {"unit": "fraction", "label": "Avg. visible transmittance", "log": False, "method": "UV-vis transmittance"},
    "sheet_resistance": {"unit": "ohm/sq", "label": "Sheet resistance", "log": True, "method": "four-point probe"},
    "thickness": {"unit": "um", "label": "Film thickness", "log": False, "method": "X-ray fluorescence"},
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------- records


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    method: str
    measured_at_c: float | None = Field(None, description="Measurement temperature, °C")

    @model_validator(mode="after")
    def _finite(self) -> "Measurement":
        if not math.isfinite(self.value):
            raise ValueError("measurement value must be finite")
        return self


class SynthesisConditions(BaseModel):
    """How the film was made. Library-level in HTEM: every sample on a library shares these."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(description="e.g. 'combinatorial RF sputtering'")
    chamber: str | None = None
    deposition_temp_c: float | None = None
    growth_pressure_mtorr: float | None = None
    base_pressure_mtorr: float | None = None
    deposition_time_min: float | None = None
    targets: list[str] = Field(default_factory=list, description="Sputter targets / precursor compounds")
    target_power_w: list[float | None] = Field(default_factory=list)
    gases: list[str] = Field(default_factory=list)
    gas_flow_sccm: list[float | None] = Field(default_factory=list)
    substrate: str | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Where the data came from, e.g. 'HTEM DB (NREL/NLR)'")
    source_sample_id: str
    source_library_id: str
    position: int | None = Field(None, description="Spot on the library (1–44 in HTEM)")
    xy_mm: list[float] | None = None
    sample_date: str | None = None
    url: str | None = None
    adapter: str
    imported_at: str = Field(default_factory=utcnow)


class ExperimentRecord(BaseModel):
    """One sample: what it is, how it was made, what was measured, and where the data came from."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    system: str = Field(description="Element system, sorted and dash-joined, e.g. 'In-O-Sn-Zn'")
    composition: dict[str, float] = Field(description="Cation (metal) atomic fractions, summing to 1")
    synthesis: SynthesisConditions
    measurements: dict[str, Measurement] = Field(default_factory=dict)
    missing: dict[str, str] = Field(default_factory=dict, description="Property → why it has no value")
    provenance: Provenance

    @field_validator("composition")
    @classmethod
    def _fractions(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("composition is empty")
        bad = [e for e in v if e not in SYMBOLS]
        if bad:
            raise ValueError(f"not element symbols: {bad}")
        if any(x < 0 for x in v.values()):
            raise ValueError("composition fractions cannot be negative")
        s = sum(v.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"composition fractions sum to {s:.4f}, not 1")
        return v

    @model_validator(mode="after")
    def _canonical_units(self) -> "ExperimentRecord":
        for name, m in self.measurements.items():
            if name not in PROPERTIES:
                raise ValueError(f"unknown property {name!r}")
            if m.unit != PROPERTIES[name]["unit"]:
                raise ValueError(f"{name} must be stored in {PROPERTIES[name]['unit']}, got {m.unit}")
        return self

    def value(self, prop: str) -> float | None:
        m = self.measurements.get(prop)
        return None if m is None else m.value


# ----------------------------------------------------------------------------- import reports


class RecordIssue(BaseModel):
    source_sample_id: str | None
    level: Literal["rejected", "dropped_measurement", "warning"]
    field: str
    reason: str


class ImportReport(BaseModel):
    """What happened to every record of one library on its way in."""

    source: str
    source_library_id: str
    system: str
    adapter: str
    received: int = 0
    accepted: int = 0
    rejected: int = 0
    measurements: dict[str, int] = Field(default_factory=dict, description="Accepted values per property")
    issues: list[RecordIssue] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)

    def reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.issues:
            key = f"{i.level}: {i.reason.split(' (')[0]}"
            out[key] = out.get(key, 0) + 1
        return out


# ------------------------------------------------------------------------------- campaign spec


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property: PropertyName
    direction: Literal["max", "min"] = "max"


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property: PropertyName
    op: Literal[">=", "<="]
    value: float

    def ok(self, v: float | None) -> bool:
        if v is None:
            return False
        return v >= self.value if self.op == ">=" else v <= self.value

    def text(self) -> str:
        return f"{PROPERTIES[self.property]['label'].lower()} {self.op} {self.value:g} {PROPERTIES[self.property]['unit']}"


class Window(BaseModel):
    """Hit a target range, e.g. band gap between 1.4 and 1.6 eV."""

    model_config = ConfigDict(extra="forbid")

    property: PropertyName
    low: float
    high: float

    @model_validator(mode="after")
    def _order(self) -> "Window":
        if self.low >= self.high:
            raise ValueError("window low must be below high")
        return self


CampaignKind = Literal["maximize", "constrained", "window", "pareto"]


class CampaignSpec(BaseModel):
    """What a partner wants from one material system, validated before anything runs."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    system: str = Field(description="Element system, e.g. 'In-O-Sn-Zn'")
    plain_text: str | None = None
    objectives: list[Objective] = Field(default_factory=list, max_length=2)
    constraints: list[Constraint] = Field(default_factory=list)
    window: Window | None = None
    budget: int = Field(60, ge=5, le=1000, description="Experiments the partner will pay for")
    batch_size: int = Field(5, ge=1, le=50)
    initial: int = Field(10, ge=1, le=100, description="Random seed experiments before the model takes over")
    strategy: Literal["bayes", "random", "grid"] = "bayes"
    surrogate: Literal["gp", "rf"] = "gp"
    failure_rate: float = Field(0.05, ge=0.0, le=0.9, description="Simulated random failures on top of real missing data")
    delay_rounds: int = Field(0, ge=0, le=5, description="Rounds before a batch's results come back")
    cost_per_experiment: float = Field(1500.0, ge=0, description="For the savings estimate, in the partner's currency")
    seed: int = 0

    @field_validator("system")
    @classmethod
    def _system(cls, v: str) -> str:
        els = v.split("-")
        bad = [e for e in els if e not in SYMBOLS]
        if bad:
            raise ValueError(f"system has unknown elements {bad}")
        return "-".join(sorted(els))

    @model_validator(mode="after")
    def _shape(self) -> "CampaignSpec":
        if self.window is None and not self.objectives:
            raise ValueError("give an objective (e.g. maximize conductivity) or a target window")
        if self.window is not None and self.objectives:
            raise ValueError("a window campaign has no separate objective")
        if len(self.objectives) == 2 and self.constraints:
            raise ValueError("two-objective campaigns don't take constraints in this prototype")
        if self.initial > self.budget:
            raise ValueError("initial experiments cannot exceed the budget")
        if self.strategy == "bayes" and self.surrogate == "rf" and len(self.objectives) == 2:
            raise ValueError("the random-forest surrogate supports one objective")
        return self

    @property
    def kind(self) -> CampaignKind:
        if self.window is not None:
            return "window"
        if len(self.objectives) == 2:
            return "pareto"
        return "constrained" if self.constraints else "maximize"

    def properties_needed(self) -> list[str]:
        props = [o.property for o in self.objectives] + [c.property for c in self.constraints]
        if self.window:
            props.append(self.window.property)
        return list(dict.fromkeys(props))

    def summary(self) -> str:
        if self.window:
            w = self.window
            goal = f"{PROPERTIES[w.property]['label'].lower()} in {w.low:g}–{w.high:g} {PROPERTIES[w.property]['unit']}"
        else:
            goal = " and ".join(f"{o.direction}imize {PROPERTIES[o.property]['label'].lower()}" for o in self.objectives)
        if self.constraints:
            goal += " with " + ", ".join(c.text() for c in self.constraints)
        return f"{self.system}: {goal}; budget {self.budget}, {self.batch_size}/round, {self.strategy}"
