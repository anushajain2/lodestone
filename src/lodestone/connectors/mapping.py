"""Column mappings: tell Lodestone which column of a lab's CSV export holds what, without code changes.

A mapping is a small YAML (or JSON) file::

    name: My lab, XRF + 4PP export
    system: In-O-Sn-Zn            # element system, anions included
    columns:
      sample_id: Sample
      composition: {In: "In at%", Sn: "Sn at%", Zn: "Zn at%"}   # or one column "In:45.7;Zn:35.5"
      deposition_temperature: {column: Tsub, unit: degC}          # unit only needed if the header has none
      conductivity: "Sigma [S/cm]"
      band_gap: Eg (eV)

Blends whose inputs are named ingredients rather than elements (a polymer and nanotube types, say) use
``composition_kind: components``; their system becomes ``blend:<sorted ingredient names>``. An export
with no sample id column can set ``row_ids: true`` (ids become row-1, row-2, ...), and repeated
measurements of the same recipe can be merged with ``replicates: average``.

A column can be named with or without the unit written in its header: ``Rho`` matches the header
``Rho [mOhm*cm]`` and the unit is read from the brackets (or parentheses). A unit given in the mapping
is used when the header has none. Units are converted with ``pint`` by the adapter; see
:data:`FIELDS` for what each field means and the dimension its unit must have.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lodestone import config
from lodestone.schemas import _COMPONENT, BLEND_PREFIX, SYMBOLS, is_blend, normalize_system

# field → (what it is, a unit with the dimension the column must have; None = not a quantity)
FIELDS: dict[str, tuple[str, str | None]] = {
    "sample_id": ("unique sample id (letters, digits, '-' and '_')", None),
    "library": ("library / batch / wafer the sample belongs to (optional)", None),
    "position": ("spot number on the library (optional, integer)", None),
    "date": ("sample date (optional, read with date_format)", None),
    "composition": ("cation composition: one 'El:value;El:value' column, or one column per element", None),
    "deposition_temperature": ("substrate / deposition temperature", "kelvin"),
    "pressure": ("growth / process pressure", "Pa"),
    "thickness": ("film thickness", "m"),
    "conductivity": ("electrical conductivity", "S/m"),
    "resistivity": ("electrical resistivity (converted to conductivity)", "ohm*m"),
    "sheet_resistance": ("sheet resistance", "ohm"),
    "band_gap": ("optical band gap", "eV"),
    "optical_edge": ("absorption-edge wavelength (converted to a band gap)", "m"),
    "transmittance": ("average visible transmittance ('%' or 'fraction')", "dimensionless"),
}
TEMPLATE_DIR = config.DATA_DIR / "templates"
HELIX_MAPPING = config.DATA_DIR / "partner_lab" / "helix_mapping.yaml"


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    unit: str | None = Field(None, description="Used when the header has no [unit]")


def _spec(v):
    return {"column": v} if isinstance(v, str) else v


class ColumnMapping(BaseModel):
    """Which column holds what. Only ``sample_id`` and ``composition`` are required."""

    model_config = ConfigDict(extra="forbid")

    name: str = "unnamed mapping"
    system: str | None = Field(None, description="Element system incl. anions, e.g. 'Co-Ni-O-Zn'")
    columns: dict[str, ColumnSpec]
    composition: dict[str, str] | None = Field(None, description="Element → column, when composition is one column per element")
    composition_unit: Literal["percent", "fraction"] = "percent"
    date_format: str = "%Y-%m-%d"
    default_library: str = Field("export", description="Library id when no library column is mapped")
    synthesis_method: str = Field("thin-film deposition (lab export)", description="How the films were made, for provenance")
    composition_kind: Literal["elements", "components"] = Field("elements", description="'components' for blends of named ingredients")
    row_ids: bool = Field(False, description="Number the rows (row-1, row-2, ...) when the export has no sample id column")
    replicates: Literal["keep", "average"] = Field("keep", description="'average' merges rows with the same recipe into one sample")

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data):
        """Accept ``field: "Header"`` and ``composition: {El: "Header", ...}`` under ``columns``."""
        if not isinstance(data, dict) or not isinstance(data.get("columns"), dict):
            return data
        data = dict(data)
        cols = dict(data["columns"])
        comp = cols.get("composition")
        if isinstance(comp, dict) and "column" not in comp:
            data["composition"] = {str(k): str(v) for k, v in comp.items()}
            cols.pop("composition")
        data["columns"] = {k: _spec(v) for k, v in cols.items() if v is not None}
        return data

    @field_validator("columns")
    @classmethod
    def _known_fields(cls, v: dict[str, ColumnSpec]) -> dict[str, ColumnSpec]:
        unknown = sorted(set(v) - set(FIELDS))
        if unknown:
            raise ValueError(f"unknown field(s) {unknown}; known fields: {sorted(FIELDS)}")
        return v

    @field_validator("system")
    @classmethod
    def _system(cls, v: str | None) -> str | None:
        return v if v is None else normalize_system(v)

    @model_validator(mode="after")
    def _shape(self) -> "ColumnMapping":
        if "sample_id" not in self.columns and not self.row_ids:
            raise ValueError("map the sample id column: columns.sample_id (or set row_ids: true)")
        if ("composition" in self.columns) == bool(self.composition):
            raise ValueError("map composition either as one column or as one column per element, not both or neither")
        if self.composition_kind == "components":
            if not self.composition:
                raise ValueError("composition_kind: components needs one column per ingredient under columns.composition")
            bad = [n for n in self.composition if not _COMPONENT.fullmatch(n)]
            if bad:
                raise ValueError(f"ingredient names may use letters, digits, '_', '+' and '.', got {bad}")
            blend = normalize_system(BLEND_PREFIX + "-".join(self.composition))
            if self.system is None:
                self.system = blend
            elif self.system != blend:
                raise ValueError(f"system {self.system!r} doesn't match the mapped ingredients ({blend})")
        else:
            if self.system is not None and is_blend(self.system):
                raise ValueError("a blend: system needs composition_kind: components")
            if self.composition:
                bad = [e for e in self.composition if e not in SYMBOLS]
                if bad:
                    raise ValueError(f"composition keys must be element symbols, got {bad} (for named ingredients set composition_kind: components)")
        if "band_gap" in self.columns and "optical_edge" in self.columns:
            raise ValueError("map either band_gap or optical_edge, not both")
        return self

    def describe(self) -> list[tuple[str, str]]:
        """(field, column[ + unit]) rows for reports."""
        rows = [(f, s.column + (f" (unit: {s.unit})" if s.unit else "")) for f, s in self.columns.items()]
        if self.composition:
            rows.append(("composition", ", ".join(f"{el} ← {c}" for el, c in self.composition.items()) + f" ({self.composition_unit})"))
        return rows


def load_mapping(path: str | Path) -> ColumnMapping:
    """Read a mapping from YAML or JSON. Errors say which key is wrong."""
    p = Path(path)
    text = p.read_text()
    data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{p.name}: a mapping must be a YAML/JSON object with a 'columns' section")
    try:
        return ColumnMapping(**data)
    except ValueError as e:
        raise ValueError(f"{p.name}: {e}") from None


def helix_mapping() -> ColumnMapping:
    """The mapping for the Helix demo lab's export (data/partner_lab/helix_second_lab_export.csv)."""
    return load_mapping(HELIX_MAPPING)


# --------------------------------------------------------------------------- header matching

_UNIT_IN_HEADER = re.compile(r"^(?P<name>.+?)\s*[\[(](?P<unit>[^\])]+)[\])]\s*$")


def split_header(header: str) -> tuple[str, str | None]:
    """'Rho [mOhm*cm]' → ('Rho', 'mOhm*cm'); 'Eg (eV)' → ('Eg', 'eV'); 'Library' → ('Library', None)."""
    m = _UNIT_IN_HEADER.match(header.strip())
    return (m.group("name").strip(), m.group("unit").strip()) if m else (header.strip(), None)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def find_header(column: str, headers: list[str]) -> str | None:
    """Exact header, else the header whose name (without its unit) matches; case- and space-insensitive."""
    for h in headers:
        if _norm(h) == _norm(column):
            return h
    matches = [h for h in headers if _norm(split_header(h)[0]) == _norm(column)]
    return matches[0] if len(matches) == 1 else None
