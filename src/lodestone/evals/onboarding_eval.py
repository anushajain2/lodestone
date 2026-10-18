"""Data onboarding and provenance evaluations.

* Import reports for every library in the HTEM snapshot: how much came in clean, what was rejected
  or dropped, and why.
* A test set of deliberately broken records (both formats) with the correct outcome known.
* Provenance spot-checks: pick random records, confirm every number matches the raw source record,
  and optionally re-fetch the sample from the live HTEM API.
"""

from __future__ import annotations

import copy
import json
import random

from lodestone import config
from lodestone.connectors import htem
from lodestone.connectors.adapters import HTEMAdapter, LabCSVAdapter, to_partner_csv

OUT = config.REPORTS_DIR / "onboarding"


def broken_cases() -> list[dict]:
    """(name, format, mutate, expectation) for deliberately broken inputs."""
    return [
        {"name": "negative conductivity", "fmt": "htem", "field": "fpm_conductivity", "value": -12.0, "expect": "dropped_measurement"},
        {"name": "band gap of 40 eV", "fmt": "htem", "field": "opt_direct_bandgap", "value": 40.0, "expect": "dropped_measurement"},
        {"name": "transmittance above 100%", "fmt": "htem", "field": "opt_average_vis_trans", "value": 1.4, "expect": "dropped_measurement"},
        {"name": "conductivity disagrees with resistivity", "fmt": "htem", "field": "fpm_resistivity", "value": 5.0, "expect": "dropped_measurement"},
        {"name": "missing composition", "fmt": "htem", "field": "xrf_concentration", "value": None, "expect": "rejected"},
        {"name": "composition sums to 250%", "fmt": "htem", "field": "xrf_concentration", "value": "x2.5", "expect": "rejected"},
        {"name": "duplicate sample id", "fmt": "htem", "field": "duplicate", "value": None, "expect": "rejected"},
        {"name": "sample from another library", "fmt": "htem", "field": "sample_library_id", "value": 1, "expect": "rejected"},
        {"name": "no four-point-probe value", "fmt": "htem", "field": "fpm_conductivity+fpm_resistivity", "value": None, "expect": "accepted_missing"},
        {"name": "clean record", "fmt": "htem", "field": None, "value": None, "expect": "accepted"},
        {"name": "unknown unit in header", "fmt": "csv", "field": "header", "value": ("Rho [mOhm*cm]", "Rho [mOhm*cmm]"), "expect": "rejected"},
        {"name": "wrong dimension in header", "fmt": "csv", "field": "header", "value": ("Substrate T [K]", "Substrate T [mm]"), "expect": "rejected"},
        {"name": "negative resistivity", "fmt": "csv", "field": 6, "value": "-1.2", "expect": "dropped_measurement"},
        {"name": "edge in µm, not nm (gap 1200 eV)", "fmt": "csv", "field": 7, "value": "0.35", "expect": "dropped_measurement"},
        {"name": "unreadable composition", "fmt": "csv", "field": 3, "value": "Co/71;Ni:11", "expect": "rejected"},
        {"name": "invalid sample id", "fmt": "csv", "field": 0, "value": "PL 12/3", "expect": "rejected"},
        {"name": "temperature 4000 °C", "fmt": "csv", "field": 4, "value": "4273", "expect": "rejected"},
        {"name": "clean CSV row", "fmt": "csv", "field": None, "value": None, "expect": "accepted"},
    ]


def _outcome(records, issues) -> str:
    if any(i.level == "rejected" for i in issues):
        return "rejected"
    if any(i.level == "dropped_measurement" for i in issues):
        return "dropped_measurement"
    # a property this lab's format never reports is not a missing measurement
    if records and any(why != "not in this lab's export" for why in records[0].missing.values()):
        return "accepted_missing"
    return "accepted"


def run_broken_records() -> list[dict]:
    lib = next(lib for lib in htem.load_snapshot("Co-Ni-O-Zn") if len(lib.get("samples") or []) > 3)
    rows = []
    for case in broken_cases():
        if case["fmt"] == "htem":
            L = copy.deepcopy(lib)
            L["samples"] = [L["samples"][0]] + ([L["samples"][0]] if case["field"] == "duplicate" else [])
            s = L["samples"][0]
            f = case["field"]
            if f == "xrf_concentration" and case["value"] == "x2.5":
                s["xrf_concentration"] = [2.5 * x for x in s["xrf_concentration"]]
            elif f == "fpm_conductivity+fpm_resistivity":
                s["fpm_conductivity"] = s["fpm_resistivity"] = s["fpm_sheet_resistance"] = None
            elif f and f != "duplicate":
                s[f] = case["value"]
            recs, rep = HTEMAdapter().adapt_library(L, set())
            if case["field"] == "duplicate":
                got = "rejected" if rep.rejected == 1 and rep.accepted == 1 else "accepted"
            else:
                got = _outcome(recs, rep.issues)
            reason = "; ".join(i.reason for i in rep.issues if i.source_sample_id)
        else:
            recs0, _ = HTEMAdapter().adapt_library(lib, set())
            text = to_partner_csv(recs0[:1])
            header, row = text.splitlines()[:2]
            if case["field"] == "header":
                header = header.replace(*case["value"])
            elif case["field"] is not None:
                cells = row.split(",")
                cells[case["field"]] = case["value"]
                row = ",".join(cells)
            recs, reps = LabCSVAdapter().adapt_csv(header + "\n" + row + "\n", "Co-Ni-O-Zn")
            issues = [i for r in reps for i in r.issues]
            got = _outcome(recs, issues)
            reason = "; ".join(i.reason for i in issues)
        rows.append({"case": case["name"], "format": case["fmt"], "expected": case["expect"], "got": got, "correct": got == case["expect"], "reason": reason})
    return rows


def run_onboarding_eval() -> dict:
    libs_out = []
    for system in htem.available_systems():
        seen: set[str] = set()
        for lib in htem.load_snapshot(system):
            _, rep = HTEMAdapter().adapt_library(lib, seen)
            libs_out.append({"system": system, "library_id": rep.source_library_id, "received": rep.received, "accepted": rep.accepted,
                             "rejected": rep.rejected, "measurements": rep.measurements, "reasons": rep.reasons()})
    totals = {}
    for s in htem.available_systems():
        rows = [r for r in libs_out if r["system"] == s]
        reasons: dict[str, int] = {}
        for r in rows:
            for k, v in r["reasons"].items():
                reasons[k] = reasons.get(k, 0) + v
        totals[s] = {"libraries": len(rows), "libraries_with_no_samples_served": sum(r["received"] == 0 for r in rows),
                     "received": sum(r["received"] for r in rows), "accepted": sum(r["accepted"] for r in rows),
                     "rejected": sum(r["rejected"] for r in rows), "reasons": reasons}
    broken = run_broken_records()
    out = {"systems": totals, "broken_records": {"cases": len(broken), "correct": sum(b["correct"] for b in broken), "rows": broken}, "libraries": libs_out}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "onboarding.json").write_text(json.dumps(out, indent=2))
    return {k: v for k, v in out.items() if k != "libraries"}


def run_provenance_check(n: int = 20, live: bool = False, seed: int = 0) -> dict:
    """Do the numbers in standard records match the raw source record (and the live API)?"""
    rng = random.Random(seed)
    pool = []
    for system in htem.available_systems():
        for lib in htem.load_snapshot(system):
            for s in lib.get("samples") or []:
                pool.append((system, lib, s))
    picks = rng.sample(pool, min(n, len(pool)))
    fields = {"conductivity": "fpm_conductivity", "band_gap": "opt_direct_bandgap", "visible_transmittance": "opt_average_vis_trans", "thickness": "thickness"}
    rows = []
    client = htem._client() if live else None
    for system, lib, s in picks:
        L = {**lib, "samples": [s]}
        recs, rep = HTEMAdapter().adapt_library(L, set())
        row = {"system": system, "htem_sample_id": s["id"], "library_id": lib["id"], "ok": True, "checked": 0, "live_checked": 0}
        if not recs:
            row.update({"ok": None, "note": "rejected at import: " + "; ".join(i.reason for i in rep.issues)})
            rows.append(row)
            continue
        r = recs[0]
        live_rec = htem._get(client, f"sample/{s['id']}") if live else None
        for prop, raw_field in fields.items():
            if prop in r.measurements:
                row["checked"] += 1
                if abs(r.measurements[prop].value - s[raw_field]) > 1e-9 * max(1, abs(s[raw_field])):
                    row["ok"] = False
                if live_rec is not None:
                    row["live_checked"] += 1
                    if live_rec.get(raw_field) is None or abs(r.measurements[prop].value - live_rec[raw_field]) > 1e-6 * max(1, abs(live_rec[raw_field])):
                        row["ok"] = False
        row["provenance"] = {"source_sample_id": r.provenance.source_sample_id, "source_library_id": r.provenance.source_library_id,
                             "url": r.provenance.url, "methods": {p: m.method for p, m in r.measurements.items()}}
        if r.provenance.source_sample_id != str(s["id"]) or r.provenance.source_library_id != str(lib["id"]):
            row["ok"] = False
        rows.append(row)
    if client:
        client.close()
    checked = [r for r in rows if r["ok"] is not None]
    out = {"sampled": len(rows), "checked": len(checked), "all_match": all(r["ok"] for r in checked), "live": live,
           "values_checked": sum(r["checked"] for r in rows), "values_checked_live": sum(r["live_checked"] for r in rows), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "provenance.json").write_text(json.dumps(out, indent=2))
    return {k: v for k, v in out.items() if k != "rows"}
