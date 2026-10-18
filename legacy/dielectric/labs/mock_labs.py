"""Three mock labs (N1) with different capabilities, capacity, noise, turnaround and output formats.

The labs are the only code that reads the catalog's hidden "true" values. Each returns a raw file in
its own format — exactly the inconsistency the adapters (:mod:`lodestone.labs.adapters`) exist to
remove. Labs also fail samples, deliver late, and occasionally send bad data (N3, X2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from lodestone import config
from lodestone.data.catalog import load_catalog
from lodestone.schemas import LabSpec, SynthesisRoute

EPS0 = 8.8541878128e-12  # F/m
HC_EV_NM = 1239.84193  # eV·nm

LABS: dict[str, LabSpec] = {
    "helios": LabSpec(
        lab_id="helios",
        name="Helios Thin-Film Lab",
        description="ALD and reactive sputtering of oxide and nitride films; MIM capacitors for C–V and spectroscopic ellipsometry.",
        measurements=["dielectric_constant", "band_gap"],
        allowed_anions=["O", "N"],
        synthesis_method="thin film (ALD / reactive sputtering)",
        capacity_per_day=3.0,
        turnaround_days=2.0,
        k_noise_rel=0.05,
        gap_noise_eV=0.08,
        failure_rate=0.05,
        late_rate=0.05,
        bad_data_rate=0.04,
        cost_per_experiment=1.0,
        output_format="CSV, one row per sample, permittivity and gap in eV",
    ),
    "kestrel": LabSpec(
        lab_id="kestrel",
        name="Kestrel Characterization Institute",
        description="Solid-state synthesis of sintered pellets; impedance analyzer and UV-vis diffuse reflectance.",
        measurements=["dielectric_constant", "band_gap"],
        allowed_anions=None,
        synthesis_method="ceramic pellet (solid-state reaction)",
        capacity_per_day=2.0,
        turnaround_days=4.0,
        k_noise_rel=0.10,
        gap_noise_eV=0.15,
        failure_rate=0.10,
        late_rate=0.15,
        bad_data_rate=0.06,
        cost_per_experiment=0.6,
        output_format="JSON with raw capacitance (pF), pellet geometry and absorption edge (nm)",
    ),
    "atlas": LabSpec(
        lab_id="atlas",
        name="Atlas National Laboratory",
        description="Glovebox synthesis for air-sensitive halides and chalcogenides; high-precision dielectric spectroscopy.",
        measurements=["dielectric_constant", "band_gap"],
        allowed_anions=None,
        synthesis_method="glovebox solid-state / sealed ampoule",
        capacity_per_day=1.5,
        turnaround_days=3.0,
        k_noise_rel=0.03,
        gap_noise_eV=0.05,
        failure_rate=0.03,
        late_rate=0.05,
        bad_data_rate=0.02,
        cost_per_experiment=2.5,
        output_format="Semicolon-separated text, decimal commas, gap in meV",
    ),
}


def capable_labs(anion: str, measurements=("dielectric_constant", "band_gap")) -> list[LabSpec]:
    return [
        lab
        for lab in LABS.values()
        if set(measurements) <= set(lab.measurements) and (lab.allowed_anions is None or anion in lab.allowed_anions)
    ]


def estimate_costs(pool: pd.DataFrame) -> np.ndarray:
    """Relative experiment cost per material: cheapest capable lab, more elements, scarce elements."""
    out = []
    for _, row in pool.iterrows():
        labs = capable_labs(row["anion"])
        base = min(l.cost_per_experiment for l in labs) if labs else 3.0
        mult = 1.0 + 0.15 * (row["n_elements"] - 1)
        if set(row["elements"]) & set(config.SCARCE_ELEMENTS + config.RARE_EARTHS):
            mult *= 1.5
        out.append(base * mult)
    return np.array(out)


_PRECURSOR_SUFFIX = {"O": "oxide", "N": "nitride", "F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide", "S": "sulfide", "Se": "selenide"}


def synthesis_route(formula: str, anion: str, lab: LabSpec) -> SynthesisRoute:
    """A plausible 'how to make it' note so the lab doesn't need to email back (illustrative only)."""
    comp = Composition(formula)
    cations = [e.symbol for e in comp.elements if e.symbol != anion]
    suffix = _PRECURSOR_SUFFIX.get(anion, anion)
    if lab.lab_id == "helios":
        precursors = [f"{c} metal-organic ALD precursor" for c in cations] + (["O3 / H2O"] if anion == "O" else ["NH3 plasma"])
        return SynthesisRoute(
            method="atomic layer deposition, 10 nm film on TiN/Si",
            precursors=precursors,
            anneal_C=600,
            atmosphere="N2",
            sample_form="MIM capacitor (TiN/film/TiN), 100 µm pads",
            notes="post-deposition anneal 60 s RTA",
        )
    air_sensitive = anion in {"Cl", "Br", "I", "S", "Se", "Te", "P"} or any(c in {"Li", "Na", "K", "Rb", "Cs"} for c in cations)
    precursors = [f"{c} {suffix} powder (99.9%)" for c in cations] or [f"{formula} powder"]
    anneal = 450 if anion in {"Cl", "Br", "I"} else 900 if anion in {"F", "S", "Se"} else 1200
    return SynthesisRoute(
        method="solid-state reaction, sealed ampoule" if air_sensitive else "solid-state reaction, ball-milled and sintered",
        precursors=precursors,
        anneal_C=anneal,
        atmosphere="Ar glovebox" if air_sensitive else "air",
        sample_form="sintered pellet, 13 mm diameter, ~0.5 mm thick, sputtered Au electrodes",
        notes="handle under inert atmosphere" if air_sensitive else "",
    )


@dataclass
class PlannedOutcome:
    """Decided at submission (seeded), revealed when the lab reports."""

    kind: str  # ok | failed | bad_data
    extra_delay_days: float
    bad_kind: str | None = None
    failure_reason: str | None = None


_FAILURES = {
    "helios": ["film delaminated during anneal", "pinholes shorted every capacitor pad", "precursor delivery fault; no film grown"],
    "kestrel": ["pellet cracked during sintering", "secondary phases dominate XRD; sample rejected", "electrode adhesion failed"],
    "atlas": ["sample decomposed during transfer", "ampoule breached; sample oxidized", "impedance drift, measurement aborted"],
}
_BAD_KINDS = {
    "helios": ["negative_k", "missing_gap", "wrong_sample", "unit_mixup"],
    "kestrel": ["zero_thickness", "missing_edge", "wrong_sample", "edge_in_ev"],
    "atlas": ["gap_in_ev_not_mev", "missing_column", "wrong_sample", "non_numeric"],
}


def plan_outcome(lab: LabSpec, rng: np.random.Generator) -> PlannedOutcome:
    u = rng.random()
    delay = float(rng.uniform(2, 6)) if rng.random() < lab.late_rate else 0.0
    if u < lab.failure_rate:
        return PlannedOutcome("failed", delay, failure_reason=str(rng.choice(_FAILURES[lab.lab_id])))
    if u < lab.failure_rate + lab.bad_data_rate:
        return PlannedOutcome("bad_data", delay, bad_kind=str(rng.choice(_BAD_KINDS[lab.lab_id])))
    return PlannedOutcome("ok", delay)


def measure(material_id: str, lab: LabSpec, rng: np.random.Generator) -> tuple[float, float]:
    """The lab 'measures' the hidden truth with its own noise."""
    row = load_catalog().loc[material_id]
    k = float(row["k_total"]) * (1.0 + lab.k_noise_rel * rng.standard_normal())
    gap = float(row["band_gap"]) + lab.gap_noise_eV * rng.standard_normal()
    return max(k, 1.05), max(gap, 0.0)


def render_raw(lab_id: str, request: dict, outcome: PlannedOutcome, k: float, gap: float, day: float, rng: np.random.Generator) -> tuple[str, str]:
    """Produce (filename, content) in the lab's native format."""
    rid, formula = request["request_id"], request["formula"]
    sample = rid if outcome.bad_kind != "wrong_sample" else rid[:-2] + "xx"
    if lab_id == "helios":
        thick = round(float(rng.uniform(9.5, 10.5)), 2)
        # Newer instrument firmware exports capacitance density instead of permittivity.
        v2 = rng.random() < 0.35
        header = (
            "Sample,Formula,C/A (nF/cm2),Eg (eV),Freq (kHz),Thickness (nm),Status,Note"
            if v2
            else "Sample,Formula,Eps_r,Eg (eV),Freq (kHz),Thickness (nm),Status,Note"
        )
        if outcome.kind == "failed":
            line = f"{sample},{formula},,,,{thick},FAILED,{outcome.failure_reason}"
        else:
            eps = f"{k * EPS0 / (thick * 1e-9) * 1e5:.2f}" if v2 else f"{k:.2f}"  # F/m² → nF/cm²
            eg = f"{gap:.3f}"
            if outcome.bad_kind == "negative_k":
                eps = "-" + eps
            elif outcome.bad_kind == "missing_gap":
                eg = ""
            elif outcome.bad_kind == "unit_mixup":
                eg = f"{HC_EV_NM / max(gap, 0.1):.1f}"  # wavelength in nm typed into the eV column
            line = f"{sample},{formula},{eps},{eg},100,{thick},OK,"
        return f"helios_{rid}.csv", header + "\n" + line + "\n"

    if lab_id == "kestrel":
        area_mm2 = round(np.pi * 6.5**2, 2)
        thick_um = round(float(rng.uniform(450, 550)), 1)
        cap_pF = k * EPS0 * (area_mm2 * 1e-6) / (thick_um * 1e-6) * 1e12
        edge_nm = HC_EV_NM / gap if gap > HC_EV_NM / 2500.0 else None  # NIR detector ends at 2500 nm
        doc: dict = {
            "lab": "Kestrel Characterization Institute",
            "specimen": sample,
            "compound": formula,
            "day": round(day, 2),
            "instrument": {"impedance": "Keysight E4990A @ 1 MHz", "optical": "UV-vis-NIR DRS, Tauc analysis"},
        }
        if outcome.kind == "failed":
            doc.update({"status": "failed", "reason": outcome.failure_reason})
        else:
            doc.update(
                {
                    "status": "complete",
                    "pellet": {"electrode_area_mm2": area_mm2, "thickness_um": thick_um},
                    "capacitance_pF": round(cap_pF, 4),
                    "absorption_edge_nm": round(edge_nm, 1) if edge_nm else ">2500",
                }
            )
            if outcome.bad_kind == "zero_thickness":
                doc["pellet"]["thickness_um"] = 0
            elif outcome.bad_kind == "missing_edge":
                doc.pop("absorption_edge_nm")
            elif outcome.bad_kind == "edge_in_ev":
                doc["absorption_edge_nm"] = round(gap, 3)  # gap in eV written into the nm field
        return f"kestrel_{rid}.json", json.dumps(doc, indent=2)

    if lab_id == "atlas":
        lines = [
            "# Atlas National Laboratory - Dielectric & Optical Characterisation",
            f"# report_day;{day:.1f}".replace(".", ","),
            "sample_ref;formula;eps_rel;Eg_meV;method;remark",
        ]
        if outcome.kind == "failed":
            lines.append(f"{sample};{formula};;;;NO DATA - {outcome.failure_reason}")
        else:
            eps = f"{k:.3f}".replace(".", ",")
            eg = f"{gap * 1000:.0f}"
            if outcome.bad_kind == "gap_in_ev_not_mev":
                eg = f"{gap:.3f}".replace(".", ",")
            elif outcome.bad_kind == "non_numeric":
                eps = "n/a (drift)"
            if outcome.bad_kind == "missing_column":
                lines[-1] = "sample_ref;formula;Eg_meV;method;remark"
                lines.append(f"{sample};{formula};{eg};broadband dielectric spectroscopy + ellipsometry;")
            else:
                lines.append(f"{sample};{formula};{eps};{eg};broadband dielectric spectroscopy + ellipsometry;")
        return f"atlas_{rid}.txt", "\n".join(lines) + "\n"

    raise ValueError(f"unknown lab {lab_id}")
