"""Adapters and validation (D2–D4): lab formats in, standard provenance-tracked records out.

Two "labs", two formats, one record schema:

* :class:`HTEMAdapter` reads HTEM DB API records (library + samples).
* :class:`LabCSVAdapter` reads a partner lab's CSV export: different column names, units written in
  the headers (resistivity in mΩ·cm, substrate temperature in K, pressure in Pa, band gap as an
  absorption-edge wavelength in nm, transmittance in %, thickness in nm, day-first dates). Units are
  parsed and converted with ``pint``.

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
from datetime import datetime

import pint

from lodestone.connectors import htem
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
                    raise Reject(f"duplicate sample id {sid}")
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


CSV_COLUMNS = {
    # their header (unit in brackets) → our raw field
    "SampleID": "sample_id",
    "Library": "library",
    "Spot": "position",
    "Composition [at.%]": "composition",
    "Substrate T [K]": "temp",
    "Process pressure [Pa]": "pressure",
    "Rho [mOhm*cm]": "resistivity",
    "Optical edge [nm]": "edge",
    "T_vis [%]": "transmittance",
    "Thickness [nm]": "thickness",
    "Date": "date",
}
_HEADER = re.compile(r"^(?P<name>[^\[]+?)\s*\[(?P<unit>[^\]]+)\]$")


def _unit_of(header: str) -> tuple[str, str | None]:
    m = _HEADER.match(header.strip())
    return (m.group("name").strip(), m.group("unit").strip()) if m else (header.strip(), None)


class LabCSVAdapter:
    """A second lab's export: same physics, different names, units and conventions."""

    name = "partner-csv/1.0"
    source = "Partner lab CSV export"

    expected_units = {  # quantity name → a unit with the dimension that column must have
        "Substrate T": "kelvin",
        "Process pressure": "Pa",
        "Rho": "ohm*m",
        "Optical edge": "m",
        "Thickness": "m",
    }
    unit_aliases = {"C": "degC", "°C": "degC", "K": "kelvin", "F": "degF"}

    def _quantity(self, value: str | None, unit: str, name: str):
        if value is None or value.strip() == "":
            return None
        try:
            magnitude = float(value.replace(",", "."))
        except ValueError:
            raise Reject(f"non-numeric {name} {value!r}") from None
        try:
            u = ureg.Unit(self.unit_aliases.get(unit, unit.replace("Ohm", "ohm").replace("Ω", "ohm").replace("µ", "u")))
        except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError, ValueError, TypeError, AttributeError):
            raise Reject(f"unknown unit '{unit}' in column '{name}'") from None
        ref = self.expected_units.get(name)
        if ref and u.dimensionality != ureg.Unit(ref).dimensionality:
            raise Reject(f"column '{name}' has unit '{unit}', which has the wrong dimension (expected something like {ref})")
        return Q_(magnitude, u)

    def adapt_csv(self, text: str, system: str, seen_ids: set[str] | None = None) -> tuple[list[ExperimentRecord], list[ImportReport]]:
        seen = set() if seen_ids is None else seen_ids
        reader = csv.DictReader(io.StringIO(text))
        headers = {_unit_of(h)[0]: h for h in (reader.fieldnames or [])}
        required = ["SampleID", "Library", "Composition"]
        missing_cols = [c for c in required if c not in headers]
        if missing_cols:
            raise Reject(f"export is missing required columns: {missing_cols}")
        reports: dict[str, ImportReport] = {}
        records: list[ExperimentRecord] = []
        for row in reader:
            lid = (row.get(headers["Library"]) or "").strip() or "unknown"
            rep = reports.setdefault(lid, ImportReport(source=self.source, source_library_id=lid, system=system, adapter=self.name))
            rep.received += 1
            sid = (row.get(headers["SampleID"]) or "").strip()
            issues: list[RecordIssue] = []
            try:
                if not re.fullmatch(r"[A-Za-z0-9_\-]+", sid or ""):
                    raise Reject(f"invalid sample id {sid!r}")
                if sid in seen:
                    raise Reject(f"duplicate sample id {sid}")
                comp_txt = row.get(headers["Composition"]) or ""
                parts = [p.split(":") for p in comp_txt.split(";") if p.strip()]
                if not parts or any(len(p) != 2 for p in parts):
                    raise Reject(f"unreadable composition {comp_txt!r} (expected 'In:45.7;Zn:35.5')")
                comp = cation_fractions([p[0].strip() for p in parts], [float(p[1]) for p in parts])

                def col(name):
                    h = headers.get(name)
                    if h is None:
                        return None
                    _, unit = _unit_of(h)
                    return self._quantity(row.get(h), unit, name) if unit else row.get(h)

                temp = col("Substrate T")
                temp_c = None if temp is None else float(temp.to(ureg.degC).magnitude)
                if temp_c is not None and not (TEMP_LIMITS_C[0] <= temp_c <= TEMP_LIMITS_C[1]):
                    raise Reject(f"substrate temperature {temp_c:.0f} °C is outside {TEMP_LIMITS_C}")
                pressure = col("Process pressure")
                rho = col("Rho")
                edge = col("Optical edge")
                tvis = row.get(headers.get("T_vis", ""), None) if "T_vis" in headers else None
                thick = col("Thickness")
                sigma = None
                if rho is not None:
                    rho_ohm_cm = rho.to("ohm*cm").magnitude
                    if rho_ohm_cm <= 0:
                        issues.append(RecordIssue(source_sample_id=sid, level="dropped_measurement", field="conductivity",
                                                  reason=f"non-positive resistivity {rho.magnitude:g} {rho.units:~}"))
                    else:
                        sigma = 1.0 / rho_ohm_cm
                gap = None
                if edge is not None:
                    nm = edge.to("nm").magnitude
                    gap = EV_NM / nm if nm > 0 else -1.0
                raw = {
                    "conductivity": sigma,
                    "band_gap": gap,
                    "visible_transmittance": None if tvis in (None, "") else float(tvis) / 100.0,
                    "thickness": None if thick is None else float(thick.to("um").magnitude),
                }
                date = (row.get(headers["Date"]) or "").strip() if "Date" in headers else ""
                iso = datetime.strptime(date, "%d/%m/%Y").date().isoformat() if date else None
                synthesis = SynthesisConditions(
                    method="combinatorial sputtering (partner lab)",
                    deposition_temp_c=temp_c,
                    growth_pressure_mtorr=None if pressure is None else float(pressure.to("mtorr").magnitude),
                )
                rec = build_record(
                    system=system, sid=sid, lid=lid, composition=comp, synthesis=synthesis, raw=raw, method_temp_c=None,
                    provenance={"source": self.source, "source_sample_id": sid, "source_library_id": lid,
                                "position": int(row[headers["Spot"]]) if "Spot" in headers and row.get(headers["Spot"]) else None,
                                "sample_date": iso, "url": None},
                    adapter=self.name, issues=issues,
                    missing_reasons={"conductivity": "no resistivity reported", "band_gap": "no optical edge reported",
                                     "visible_transmittance": "no transmittance reported", "thickness": "no thickness reported",
                                     "sheet_resistance": "not in this lab's export"},
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
    w.writerow(list(CSV_COLUMNS))
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
