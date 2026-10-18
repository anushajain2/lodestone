"""Lab adapters (X2): turn each lab's native output into the one standard result schema.

Each adapter parses its lab's format, converts units, and then everything goes through the same
validation (the :class:`~lodestone.schemas.ExperimentResult` contract plus a check that the file
is about the sample that was requested). Anything that fails is rejected with a reason a lab
operator can act on — never silently dropped, never silently "fixed".
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import dataclass

from pydantic import ValidationError

from lodestone.labs.mock_labs import EPS0, HC_EV_NM, LABS
from lodestone.schemas import ExperimentResult


class AdapterError(ValueError):
    """The raw file cannot be turned into a valid result. The message is the rejection reason."""


@dataclass
class ParsedRow:
    sample_ref: str
    formula: str | None
    status: str  # ok | failed
    k: float | None = None
    gap_eV: float | None = None
    k_method: str = ""
    gap_method: str = ""
    failure_reason: str | None = None
    conversions: str = ""


@dataclass
class AdapterOutcome:
    accepted: bool
    result: ExperimentResult | None
    reason: str | None
    conversions: str = ""


def _num(text: str, field: str) -> float:
    t = (text or "").strip()
    if t == "":
        raise AdapterError(f"missing value for {field}")
    try:
        return float(t.replace(",", "."))
    except ValueError:
        raise AdapterError(f"non-numeric value {t!r} for {field}") from None


# --------------------------------------------------------------------------------------- helios


def parse_helios(content: str) -> ParsedRow:
    rows = list(csv.DictReader(io.StringIO(content)))
    if len(rows) != 1:
        raise AdapterError(f"expected exactly one sample row, found {len(rows)}")
    r = {k.strip(): (v or "").strip() for k, v in rows[0].items()}
    if r.get("Status", "").upper() == "FAILED":
        return ParsedRow(r.get("Sample", ""), r.get("Formula"), "failed", failure_reason=r.get("Note") or "lab marked sample FAILED")
    conv = ""
    if "Eps_r" in r:
        k = _num(r["Eps_r"], "Eps_r")
    elif "C/A (nF/cm2)" in r:
        ca = _num(r["C/A (nF/cm2)"], "C/A (nF/cm2)")
        d_nm = _num(r.get("Thickness (nm)", ""), "Thickness (nm)")
        if d_nm <= 0:
            raise AdapterError("film thickness must be positive to convert C/A to permittivity")
        k = ca * 1e-5 * (d_nm * 1e-9) / EPS0  # nF/cm² → F/m², then k = (C/A)·d/ε0
        conv = f"k from capacitance density {ca:g} nF/cm² and thickness {d_nm:g} nm"
    else:
        raise AdapterError("no permittivity column (expected 'Eps_r' or 'C/A (nF/cm2)')")
    gap = _num(r.get("Eg (eV)", ""), "Eg (eV)")
    return ParsedRow(
        r.get("Sample", ""), r.get("Formula"), "ok", k, gap,
        k_method="C–V on MIM capacitor @ 100 kHz", gap_method="spectroscopic ellipsometry", conversions=conv,
    )


# -------------------------------------------------------------------------------------- kestrel


def parse_kestrel(content: str) -> ParsedRow:
    try:
        doc = json.loads(content)
    except json.JSONDecodeError as e:
        raise AdapterError(f"file is not valid JSON ({e.msg})") from None
    sample = str(doc.get("specimen", ""))
    if doc.get("status") == "failed":
        return ParsedRow(sample, doc.get("compound"), "failed", failure_reason=doc.get("reason") or "lab reported failure")
    if doc.get("status") != "complete":
        raise AdapterError(f"unknown status {doc.get('status')!r}")
    pellet = doc.get("pellet") or {}
    area = _num(str(pellet.get("electrode_area_mm2", "")), "electrode_area_mm2")
    thick = _num(str(pellet.get("thickness_um", "")), "thickness_um")
    cap = _num(str(doc.get("capacitance_pF", "")), "capacitance_pF")
    if area <= 0 or thick <= 0:
        raise AdapterError(f"pellet geometry must be positive (area {area} mm², thickness {thick} µm)")
    k = cap * 1e-12 * (thick * 1e-6) / (EPS0 * area * 1e-6)
    if "absorption_edge_nm" not in doc:
        raise AdapterError("missing absorption_edge_nm")
    edge = doc["absorption_edge_nm"]
    if isinstance(edge, str) and edge.startswith(">"):
        raise AdapterError(f"absorption edge {edge} nm is beyond the detector, so the band gap is below the instrument range (< {HC_EV_NM / 2500:.2f} eV); re-route to a lab with IR ellipsometry if needed")
    edge_nm = _num(str(edge), "absorption_edge_nm")
    if not (150.0 <= edge_nm <= 2500.0):
        raise AdapterError(f"absorption edge {edge_nm:g} nm is outside the UV-vis-NIR range (150–2500 nm); likely wrong units")
    gap = HC_EV_NM / edge_nm
    return ParsedRow(
        sample, doc.get("compound"), "ok", k, gap,
        k_method="impedance @ 1 MHz on sintered pellet", gap_method="UV-vis diffuse reflectance (Tauc)",
        conversions=f"k = C·d/(ε0·A) from {cap:g} pF, {thick:g} µm, {area:g} mm²; Eg = hc/λ from {edge_nm:g} nm",
    )


# ---------------------------------------------------------------------------------------- atlas


def parse_atlas(content: str) -> ParsedRow:
    lines = [l for l in content.splitlines() if l.strip() and not l.startswith("#")]
    if len(lines) < 2:
        raise AdapterError("no data row in report")
    header = [h.strip() for h in lines[0].split(";")]
    values = [v.strip() for v in lines[1].split(";")]
    row = dict(zip(header, values + [""] * (len(header) - len(values))))
    remark = row.get("remark", "")
    if remark.upper().startswith("NO DATA"):
        return ParsedRow(row.get("sample_ref", ""), row.get("formula"), "failed", failure_reason=remark.split("-", 1)[-1].strip() or remark)
    for col in ("sample_ref", "eps_rel", "Eg_meV"):
        if col not in header:
            raise AdapterError(f"missing column '{col}' in report header")
    k = _num(row["eps_rel"], "eps_rel")
    eg_raw = row["Eg_meV"]
    eg_meV = _num(eg_raw, "Eg_meV")
    if "," in eg_raw or "." in eg_raw or eg_meV < 50:
        raise AdapterError(f"Eg_meV value {eg_raw!r} looks like eV, not meV; refusing to guess the unit")
    return ParsedRow(
        row["sample_ref"], row.get("formula"), "ok", k, eg_meV / 1000.0,
        k_method="broadband dielectric spectroscopy", gap_method="ellipsometry",
        conversions=f"decimal commas; Eg {eg_meV:g} meV → {eg_meV / 1000:.3f} eV",
    )


PARSERS = {"helios": parse_helios, "kestrel": parse_kestrel, "atlas": parse_atlas}
ADAPTER_VERSION = "adapters/1.0"


def adapt(lab_id: str, filename: str, content: str, request: dict, received_day: float) -> AdapterOutcome:
    """Parse, convert and validate one raw lab file against the request it answers."""
    lab = LABS.get(lab_id)
    if lab is None:
        return AdapterOutcome(False, None, f"unknown lab '{lab_id}'")
    try:
        parsed = PARSERS[lab_id](content)
        if parsed.sample_ref != request["request_id"]:
            raise AdapterError(
                f"sample reference {parsed.sample_ref!r} does not match request {request['request_id']!r}"
            )
        if parsed.formula and parsed.formula.replace(" ", "") != request["formula"]:
            raise AdapterError(f"formula {parsed.formula!r} does not match requested {request['formula']!r}")
        provenance = {
            "lab_id": lab.lab_id,
            "lab_name": lab.name,
            "instrument": f"{parsed.k_method} / {parsed.gap_method}".strip(" /") or "n/a",
            "measured_day": received_day,
            "received_day": received_day,
            "raw_file": filename,
            "adapter": f"{ADAPTER_VERSION}:{lab_id}",
        }
        base = {
            "result_id": f"res-{uuid.uuid4().hex[:10]}",
            "request_id": request["request_id"],
            "partner_id": request["partner_id"],
            "program_id": request["program_id"],
            "material_id": request["material_id"],
            "formula": request["formula"],
            "provenance": provenance,
        }
        if parsed.status == "failed":
            result = ExperimentResult(**base, status="failed", failure_reason=parsed.failure_reason)
        else:
            result = ExperimentResult(
                **base,
                status="ok",
                dielectric_constant={
                    "value": round(parsed.k, 4), "unit": "dimensionless",
                    "uncertainty": round(parsed.k * lab.k_noise_rel, 4), "method": parsed.k_method,
                },
                band_gap={
                    "value": round(parsed.gap_eV, 4), "unit": "eV",
                    "uncertainty": lab.gap_noise_eV, "method": parsed.gap_method,
                },
            )
        return AdapterOutcome(True, result, None, parsed.conversions)
    except AdapterError as e:
        return AdapterOutcome(False, None, str(e))
    except ValidationError as e:
        msgs = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        return AdapterOutcome(False, None, f"failed the result contract: {msgs}")
    except Exception as e:  # malformed beyond what the parser anticipates
        return AdapterOutcome(False, None, f"unreadable file ({type(e).__name__}: {e})")
