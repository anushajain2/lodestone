"""Adapters and validation (D2–D4): lab formats in, standard provenance-tracked records out.

Two "labs", two formats, one record schema:

* :class:`HTEMAdapter` reads HTEM DB API records (library + samples).
* :class:`LabCSVAdapter` reads any lab's CSV export through a column mapping (YAML/JSON, see
  :mod:`lodestone.connectors.mapping`): which column is the sample id, composition, temperature,
  resistivity or conductivity, optical edge or band gap, and so on, with units taken from the headers
  (``Rho [mOhm*cm]``) or the mapping. The Helix demo lab's export (resistivity in mΩ·cm, substrate
  temperature in K, pressure in Pa, band gap as an absorption-edge wavelength in nm, day-first dates)
  is read through ``data/partner_lab/helix_mapping.yaml``. Units are converted with ``pint``.

Validation is the same for both. A record is **rejected** (with a reason) when it can't be used at
all: no composition, compositions that don't add up, duplicate sample ids, unknown units. A single
**measurement is dropped** (with a reason) when it is impossible: negative conductivity, a band gap of
40 eV, 140% transmittance, resistivity and conductivity that disagree. Missing values are not errors:
they are recorded with why they are missing, because a lab replay must be able to say "this
measurement failed".
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime

import pint

from lodestone.connectors import htem
from lodestone.connectors.mapping import FIELDS, ColumnMapping, find_header, helix_mapping, split_header
from lodestone.schemas import ExperimentRecord, ImportReport, Measurement, PROPERTIES, Provenance, RecordIssue, SynthesisConditions

ureg = pint.UnitRegistry()
ureg.define("square = [] = sq")  # 'ohm/sq'
Q_ = ureg.Quantity

ANIONS = {"O", "N", "S", "Se", "Te", "F", "Cl", "Br", "I", "H", "C", "P"}
EV_NM = 1239.84193

# Physical sanity limits for these thin films (canonical units).
LIMITS = {
    "conductivity": (1e-8, 1e5),  # S/cm
    "sheet_resistance": (1e-3, 1e12),  # ohm/sq
    "band_gap": (0.1, 8.0),  # eV
    "visible_transmittance": (0.0, 1.0),  # fraction
    "thickness": (0.001, 10.0),  # µm
}
TEMP_LIMITS_C = (-50.0, 1200.0)


class Reject(ValueError):
    """The whole record is unusable; the message is the reason."""


def _metal(compound: str) -> str | None:
    m = re.match(r"^([A-Z][a-z]?)", compound or "")
    return m.group(1) if m else None


def cation_fractions(names: list[str], amounts: list[float]) -> dict[str, float]:
    """XRF compounds + concentrations → metal atomic fractions summing to 1."""
    if not names or not amounts or len(names) != len(amounts):
        raise Reject("no usable XRF composition (compounds and concentrations missing or mismatched)")
    out: dict[str, float] = {}
    for name, amt in zip(names, amounts):
        el = _metal(name)
        if el is None or amt is None:
            raise Reject(f"unreadable XRF entry {name!r}={amt!r}")
        if el in ANIONS:
            continue
        if amt < 0:
            raise Reject(f"negative XRF concentration for {el} ({amt})")
        out[el] = out.get(el, 0.0) + float(amt)
    total = sum(out.values())
    if total <= 0:
        raise Reject("XRF composition has no metal content")
    if not (90.0 <= total <= 110.0):
        raise Reject(f"XRF concentrations sum to {total:.1f}%, not ~100% (wrong units or truncated record)")
    return {k: v / total for k, v in sorted(out.items())}


def _check(prop: str, value: float | None, issues: list[RecordIssue], sid: str) -> float | None:
    if value is None:
        return None
    lo, hi = LIMITS[prop]
    if not math.isfinite(value) or not (lo <= value <= hi):
        unit = PROPERTIES[prop]["unit"]
        issues.append(RecordIssue(source_sample_id=sid, level="dropped_measurement", field=prop,
                                  reason=f"impossible {PROPERTIES[prop]['label'].lower()} {value:g} {unit} (allowed {lo:g}–{hi:g})"))
        return None
    return value


def build_record(
    *, system: str, sid: str, lid: str, composition: dict[str, float], synthesis: SynthesisConditions,
    raw: dict[str, float | None], method_temp_c: float | None, provenance: dict, adapter: str,
    issues: list[RecordIssue], missing_reasons: dict[str, str],
) -> ExperimentRecord:
    """Shared validation for every adapter. ``raw`` holds canonical-unit values (or None)."""
    meas: dict[str, Measurement] = {}
    missing: dict[str, str] = {}
    sigma = _check("conductivity", raw.get("conductivity"), issues, sid)
    rho_sigma = raw.get("resistivity_derived_conductivity")
    if raw.get("conductivity") is not None and sigma is None:
        rho_sigma = None  # already dropped as impossible; don't resurrect it from the resistivity
    if sigma is not None and rho_sigma is not None and abs(sigma * (1 / rho_sigma) - 1) > 0.05:
        issues.append(RecordIssue(source_sample_id=sid, level="dropped_measurement", field="conductivity",
                                  reason=f"conductivity {sigma:g} S/cm disagrees with 1/resistivity {rho_sigma:g} S/cm by more than 5%"))
        sigma = None
    elif sigma is None and rho_sigma is not None:
        sigma = _check("conductivity", rho_sigma, issues, sid)
    values = {
        "conductivity": sigma,
        "sheet_resistance": _check("sheet_resistance", raw.get("sheet_resistance"), issues, sid),
        "band_gap": _check("band_gap", raw.get("band_gap"), issues, sid),
        "visible_transmittance": _check("visible_transmittance", raw.get("visible_transmittance"), issues, sid),
        "thickness": _check("thickness", raw.get("thickness"), issues, sid),
    }
    for prop, v in values.items():
        if v is None:
            dropped = any(i.field == prop and i.source_sample_id == sid for i in issues)
            missing[prop] = "dropped by validation" if dropped else missing_reasons.get(prop, "not measured")
        else:
            meas[prop] = Measurement(
                value=float(v), unit=PROPERTIES[prop]["unit"], method=PROPERTIES[prop]["method"],
                measured_at_c=method_temp_c if prop in ("conductivity", "sheet_resistance") else None,
            )
    return ExperimentRecord(
        record_id=f"rec-{provenance['source'].split()[0].lower()}-{sid}",
        system=system,
        composition=composition,
        synthesis=synthesis,
        measurements=meas,
        missing=missing,
        provenance=Provenance(**provenance, adapter=adapter),
    )


# ---------------------------------------------------------------------------------------- HTEM


class HTEMAdapter:
    name = "htem-api/1.0"
    source = "HTEM DB (NREL/NLR)"

    def synthesis(self, lib: dict, issues: list[RecordIssue]) -> SynthesisConditions:
        t = lib.get("deposition_initial_temp_c")
        if t is not None and not (TEMP_LIMITS_C[0] <= t <= TEMP_LIMITS_C[1]):
            issues.append(RecordIssue(source_sample_id=None, level="warning", field="deposition_temp_c",
                                      reason=f"deposition temperature {t} °C is outside {TEMP_LIMITS_C}; ignored"))
            t = None
        elif t is None:
            issues.append(RecordIssue(source_sample_id=None, level="warning", field="deposition_temp_c",
                                      reason="deposition temperature not recorded for this library"))
        pulses = [p for p in (lib.get("deposition_target_pulses") or []) if p]  # HTEM stores [None, None, None] for sputtered films
        method = "combinatorial pulsed laser deposition" if pulses else "combinatorial sputtering"
        return SynthesisConditions(
            method=method,
            chamber=f"pdac-{lib.get('pdac')}" if lib.get("pdac") else None,
            deposition_temp_c=t,
            growth_pressure_mtorr=lib.get("deposition_growth_pressure_mtorr"),
            base_pressure_mtorr=lib.get("deposition_base_pressure_mtorr"),
            deposition_time_min=lib.get("deposition_sample_time_min"),
            targets=[c for c in (lib.get("deposition_compounds") or []) if c],
            target_power_w=[p for p, c in zip(lib.get("deposition_power") or [], lib.get("deposition_compounds") or []) if c],
            gases=[g for g in (lib.get("deposition_gases") or []) if g],
            gas_flow_sccm=[f for f, g in zip(lib.get("deposition_gas_flow_sccm") or [], lib.get("deposition_gases") or []) if g],
            substrate=lib.get("deposition_substrate_material"),
        )

    def adapt_library(self, lib: dict, seen_ids: set[str] | None = None) -> tuple[list[ExperimentRecord], ImportReport]:
        seen = set() if seen_ids is None else seen_ids
        system = htem.system_key(lib.get("elements"))
        lid = str(lib["id"])
        report = ImportReport(source=self.source, source_library_id=lid, system=system, adapter=self.name)
        issues: list[RecordIssue] = []
        synthesis = self.synthesis(lib, issues)
        samples = lib.get("samples") or []
        if not samples:
            issues.append(RecordIssue(source_sample_id=None, level="warning", field="samples",
                                      reason=f"library lists {len(lib.get('sample_ids') or [])} samples but the API served no sample records"))
        records = []
        for s in samples:
            report.received += 1
            sid = str(s.get("id"))
            try:
                if not s.get("id"):
                    raise Reject("sample has no id")
                if sid in seen:
                    raise Reject(f"duplicate sample id ({sid})")
                if str(s.get("sample_library_id")) != lid:
                    raise Reject(f"sample claims library {s.get('sample_library_id')}, imported under {lid}")
                comp = cation_fractions(s.get("xrf_compounds") or [], s.get("xrf_concentration") or [])
                rho = s.get("fpm_resistivity")
                raw = {
                    "conductivity": s.get("fpm_conductivity"),
                    "resistivity_derived_conductivity": (1.0 / rho) if rho else None,
                    "sheet_resistance": s.get("fpm_sheet_resistance"),
                    "band_gap": s.get("opt_direct_bandgap"),
                    "visible_transmittance": s.get("opt_average_vis_trans"),
                    "thickness": s.get("thickness"),
                }
                rec = build_record(
                    system=system, sid=sid, lid=lid, composition=comp, synthesis=synthesis, raw=raw,
                    method_temp_c=s.get("absolute_temp_c"),
                    provenance={
                        "source": self.source, "source_sample_id": sid, "source_library_id": lid,
                        "position": s.get("position"), "xy_mm": (s.get("xyz_mm") or [None, None])[:2],
                        "sample_date": lib.get("sample_date"), "url": htem.sample_url(int(sid)),
                    },
                    adapter=self.name, issues=issues,
                    missing_reasons={
                        "conductivity": "no four-point-probe value (often: film too resistive for the instrument)",
                        "sheet_resistance": "no four-point-probe value",
                        "band_gap": "no direct-gap Tauc fit in the optical data",
                        "visible_transmittance": "no optical transmittance data",
                        "thickness": "no XRF thickness",
                    },
                )
                seen.add(sid)
                records.append(rec)
            except Reject as e:
                issues.append(RecordIssue(source_sample_id=sid, level="rejected", field="record", reason=str(e)))
            except ValueError as e:  # schema contract
                issues.append(RecordIssue(source_sample_id=sid, level="rejected", field="record", reason=f"failed the record contract: {e}"))
        report.accepted = len(records)
        report.rejected = sum(1 for i in issues if i.level == "rejected")
        report.issues = issues
        for r in records:
            for p in r.measurements:
                report.measurements[p] = report.measurements.get(p, 0) + 1
        return records, report


# ------------------------------------------------------------------------------- partner CSV


# The Helix demo lab's export header (written by :func:`to_partner_csv`, read via helix_mapping.yaml).
CSV_COLUMNS = [
    "SampleID", "Library", "Spot", "Composition [at.%]", "Substrate T [K]", "Process pressure [Pa]",
    "Rho [mOhm*cm]", "Optical edge [nm]", "T_vis [%]", "Thickness [nm]", "Date",
]
UNIT_ALIASES = {"C": "degC", "°C": "degC", "K": "kelvin", "F": "degF", "°F": "degF", "%": "percent"}
# measured property → mapping fields that can supply it
PROPERTY_SOURCES = {
    "conductivity": ("conductivity", "resistivity"),
    "band_gap": ("band_gap", "optical_edge"),
    "visible_transmittance": ("transmittance",),
    "sheet_resistance": ("sheet_resistance",),
    "thickness": ("thickness",),
}


def parse_unit(text: str) -> pint.Unit:
    """Lab spellings → pint: 'mOhm*cm', 'mΩ·cm', 'µm', '°C', 'K', 'mTorr', '%'."""
    u = UNIT_ALIASES.get(text.strip(), text.strip())
    u = u.replace("Ohm", "ohm").replace("Ω", "ohm").replace("µ", "u").replace("μ", "u").replace("Torr", "torr")
    u = re.sub(r"(?<=[A-Za-z])[.·](?=[A-Za-z])", "*", u)
    return ureg.Unit(u)


@dataclass
class MappedColumn:
    """One mapped column, resolved against the file's header once. ``error`` rejects rows that use it."""

    field: str
    header: str
    unit_text: str | None = None
    unit: pint.Unit | None = None
    divisor: float = 1.0  # transmittance: % → fraction
    error: str | None = None


class LabCSVAdapter:
    """Any lab's CSV export, read through a :class:`~lodestone.connectors.mapping.ColumnMapping`.

    Without a mapping it reads the Helix demo lab's export (``data/partner_lab/helix_mapping.yaml``).
    """

    name = "partner-csv/1.0"
    source = "Partner lab CSV export"

    def __init__(self, mapping: ColumnMapping | None = None):
        self.mapping = mapping or helix_mapping()

    # ---------------------------------------------------------------- columns

    def _resolve(self, field: str, header: str, mapping_unit: str | None) -> MappedColumn:
        col = MappedColumn(field, header)
        ref = FIELDS[field][1]
        if ref is None:
            return col
        name, header_unit = split_header(header)
        col.unit_text = header_unit or mapping_unit
        if col.unit_text is None:
            col.error = f"column '{header}' ({field}) has no unit: write it in the header, e.g. '{name} [unit]', or give `unit:` in the mapping"
            return col
        if field == "transmittance":
            scales = {"%": 100.0, "percent": 100.0, "fraction": 1.0, "1": 1.0}
            if col.unit_text not in scales:
                col.error = f"unknown unit '{col.unit_text}' in column '{name}' (transmittance takes '%' or 'fraction')"
            col.divisor = scales.get(col.unit_text, 1.0)
            return col
        try:
            col.unit = parse_unit(col.unit_text)
            if header_unit and mapping_unit and parse_unit(mapping_unit) != col.unit:
                col.error = f"column '{name}' says '{header_unit}' in its header but the mapping says '{mapping_unit}'"
        except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError, ValueError, TypeError, AttributeError, AssertionError):
            col.error = f"unknown unit '{col.unit_text}' in column '{name}'"
            return col
        if col.unit.dimensionality != ureg.Unit(ref).dimensionality:
            col.error = f"column '{name}' has unit '{col.unit_text}', which has the wrong dimension (expected something like {ref})"
        return col

    def plan(self, headers: list[str]) -> tuple[dict[str, MappedColumn], dict[str, str], list[str]]:
        """Mapped field → column, element → composition column, and the file's unmapped columns.

        Raises :class:`Reject` when the mapping names a column the file doesn't have."""
        m = self.mapping
        cols: dict[str, MappedColumn] = {}
        comp: dict[str, str] = {}
        missing = []
        for f, spec in m.columns.items():
            h = find_header(spec.column, headers)
            if h is None:
                missing.append(f"{f} → '{spec.column}'")
            else:
                cols[f] = self._resolve(f, h, spec.unit)
        for el, c in (m.composition or {}).items():
            h = find_header(c, headers)
            if h is None:
                missing.append(f"composition {el} → '{c}'")
            else:
                comp[el] = h
        if missing:
            raise Reject(f"export is missing required columns: {missing} (mapping '{m.name}'; the file has {headers})")
        used = {c.header for c in cols.values()} | set(comp.values())
        return cols, comp, [h for h in headers if h not in used]

    @staticmethod
    def _quantity(row: dict, col: MappedColumn | None):
        if col is None:
            return None
        value = (row.get(col.header) or "").strip()
        if value == "":
            return None
        try:
            magnitude = float(value.replace(",", "."))
        except ValueError:
            raise Reject(f"non-numeric {col.field.replace('_', ' ')} {value!r}") from None
        if col.error:
            raise Reject(col.error)
        return magnitude / col.divisor if col.unit is None else Q_(magnitude, col.unit)

    def _composition(self, row: dict, cols: dict[str, MappedColumn], comp: dict[str, str]) -> dict[str, float]:
        scale = 100.0 if self.mapping.composition_unit == "fraction" else 1.0
        if comp:
            names, amounts = [], []
            for el, h in comp.items():
                cell = (row.get(h) or "").strip()
                try:
                    amounts.append(scale * float(cell.replace(",", ".")))
                except ValueError:
                    raise Reject(f"unreadable {el} amount {cell!r} in column '{h}'") from None
                names.append(el)
            return cation_fractions(names, amounts)
        txt = row.get(cols["composition"].header) or ""
        parts = [p.split(":") for p in txt.split(";") if p.strip()]
        try:
            if not parts or any(len(p) != 2 for p in parts):
                raise ValueError
            return cation_fractions([p[0].strip() for p in parts], [scale * float(p[1]) for p in parts])
        except ValueError as e:
            if isinstance(e, Reject):
                raise
            raise Reject(f"unreadable composition {txt!r} (expected 'In:45.7;Zn:35.5')") from None

    # ---------------------------------------------------------------- rows

    def adapt_csv(self, text: str, system: str | None = None, seen_ids: set[str] | None = None) -> tuple[list[ExperimentRecord], list[ImportReport]]:
        system = system or self.mapping.system
        if not system:
            raise Reject("which material system is this export? give it in the mapping (system: Co-Ni-O-Zn) or on the command line")
        seen = set() if seen_ids is None else seen_ids
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        cols, comp, _ = self.plan(list(reader.fieldnames or []))
        missing_reasons = {}
        for prop, fields in PROPERTY_SOURCES.items():
            have = [f.replace("_", " ") for f in fields if f in cols]
            missing_reasons[prop] = f"no {' or '.join(have)} reported" if have else "not in this lab's export"
        reports: dict[str, ImportReport] = {}
        records: list[ExperimentRecord] = []
        for row in reader:
            lid = (row.get(cols["library"].header) or "").strip() if "library" in cols else ""
            lid = lid or self.mapping.default_library
            rep = reports.setdefault(lid, ImportReport(source=self.source, source_library_id=lid, system=system, adapter=self.name))
            rep.received += 1
            sid = (row.get(cols["sample_id"].header) or "").strip()
            issues: list[RecordIssue] = []
            try:
                if not re.fullmatch(r"[A-Za-z0-9_\-]+", sid or ""):
                    raise Reject(f"invalid sample id {sid!r}")
                if sid in seen:
                    raise Reject(f"duplicate sample id ({sid})")
                composition = self._composition(row, cols, comp)
                q = {f: self._quantity(row, cols.get(f)) for f in FIELDS if FIELDS[f][1] is not None}
                temp_c = None if q["deposition_temperature"] is None else float(q["deposition_temperature"].to(ureg.degC).magnitude)
                if temp_c is not None and not (TEMP_LIMITS_C[0] <= temp_c <= TEMP_LIMITS_C[1]):
                    raise Reject(f"substrate temperature {temp_c:.0f} °C is outside {TEMP_LIMITS_C}")
                rho_sigma = None
                if (rho := q["resistivity"]) is not None:
                    rho_ohm_cm = rho.to("ohm*cm").magnitude
                    if rho_ohm_cm <= 0:
                        issues.append(RecordIssue(source_sample_id=sid, level="dropped_measurement", field="conductivity",
                                                  reason=f"non-positive resistivity {rho.magnitude:g} {rho.units:~}"))
                    else:
                        rho_sigma = 1.0 / rho_ohm_cm
                gap = None if q["band_gap"] is None else float(q["band_gap"].to("eV").magnitude)
                if (edge := q["optical_edge"]) is not None:
                    nm = edge.to("nm").magnitude
                    gap = EV_NM / nm if nm > 0 else -1.0
                raw = {
                    "conductivity": None if q["conductivity"] is None else float(q["conductivity"].to("S/cm").magnitude),
                    "resistivity_derived_conductivity": rho_sigma,
                    "sheet_resistance": None if q["sheet_resistance"] is None else float(q["sheet_resistance"].to("ohm").magnitude),
                    "band_gap": gap,
                    "visible_transmittance": q["transmittance"],
                    "thickness": None if q["thickness"] is None else float(q["thickness"].to("um").magnitude),
                }
                iso = None
                if "date" in cols and (date := (row.get(cols["date"].header) or "").strip()):
                    try:
                        iso = datetime.strptime(date, self.mapping.date_format).date().isoformat()
                    except ValueError:
                        issues.append(RecordIssue(source_sample_id=sid, level="warning", field="date",
                                                  reason=f"unreadable date, kept without one (got {date!r}, date_format is {self.mapping.date_format!r})"))
                position = None
                if "position" in cols and (pos := (row.get(cols["position"].header) or "").strip()):
                    try:
                        position = int(float(pos))
                    except ValueError:
                        issues.append(RecordIssue(source_sample_id=sid, level="warning", field="position",
                                                  reason=f"position is not a number, kept without one (got {pos!r})"))
                synthesis = SynthesisConditions(
                    method=self.mapping.synthesis_method,
                    deposition_temp_c=temp_c,
                    growth_pressure_mtorr=None if q["pressure"] is None else float(q["pressure"].to("mtorr").magnitude),
                )
                rec = build_record(
                    system=system, sid=sid, lid=lid, composition=composition, synthesis=synthesis, raw=raw, method_temp_c=None,
                    provenance={"source": self.source, "source_sample_id": sid, "source_library_id": lid,
                                "position": position, "sample_date": iso, "url": None},
                    adapter=self.name, issues=issues, missing_reasons=missing_reasons,
                )
                seen.add(sid)
                records.append(rec)
                rep.accepted += 1
                for p in rec.measurements:
                    rep.measurements[p] = rep.measurements.get(p, 0) + 1
            except Reject as e:
                issues.append(RecordIssue(source_sample_id=sid or None, level="rejected", field="record", reason=str(e)))
                rep.rejected += 1
            except ValueError as e:
                issues.append(RecordIssue(source_sample_id=sid or None, level="rejected", field="record", reason=f"failed the record contract: {e}"))
                rep.rejected += 1
            rep.issues.extend(issues)
        return records, list(reports.values())



def to_partner_csv(records: list[ExperimentRecord]) -> str:
    """Re-export standard records in the partner lab's format (used to create the second lab's data)."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(CSV_COLUMNS)
    for r in records:
        sigma = r.value("conductivity")
        gap = r.value("band_gap")
        tv = r.value("visible_transmittance")
        th = r.value("thickness")
        t = r.synthesis.deposition_temp_c
        p = r.synthesis.growth_pressure_mtorr
        date = r.provenance.sample_date
        w.writerow([
            f"PL-{r.provenance.source_sample_id}",
            f"LIB-{r.provenance.source_library_id}",
            r.provenance.position or "",
            ";".join(f"{el}:{100 * f:.2f}" for el, f in r.composition.items()),
            "" if t is None else f"{t + 273.15:.1f}",
            "" if p is None else f"{(p * ureg.mtorr).to('Pa').magnitude:.4f}",
            "" if sigma is None else f"{1000.0 / sigma:.6g}",
            "" if gap is None else f"{EV_NM / gap:.2f}",
            "" if tv is None else f"{100 * tv:.1f}",
            "" if th is None else f"{1000 * th:.1f}",
            "" if not date else datetime.fromisoformat(date[:10]).strftime("%d/%m/%Y"),
        ])
    return out.getvalue()
