"""Platform evaluations: result normalization (X2) and loop survival under lab failures (N3).

Normalization: a test set of messy lab files — every lab's format, its unit variants, failures and
each kind of bad data — with the correct outcome known. A file is handled correctly when the adapter
accepts a good file with values within tolerance of what the lab measured, records a lab failure
as a failed result, or rejects bad data with a reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lodestone import config
from lodestone.data.catalog import load_catalog
from lodestone.labs import adapters
from lodestone.labs.mock_labs import _BAD_KINDS, LABS, PlannedOutcome, capable_labs, render_raw

FILES_DIR = config.ROOT / "evals" / "messy_lab_files"


def build_messy_set(n_per_case: int = 3, seed: int = 11) -> list[dict]:
    """Generate the file set and write it to evals/messy_lab_files/ with a manifest."""
    rng = np.random.default_rng(seed)
    cat = load_catalog()
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lab_id, lab in LABS.items():
        eligible = [m for m in cat.index if lab in capable_labs(cat.loc[m, "anion"]) and cat.loc[m, "band_gap"] > 0.6]
        cases = [("ok", None)] * (n_per_case * 3) + [("failed", None)] * n_per_case + [("bad_data", b) for b in _BAD_KINDS[lab_id] for _ in range(n_per_case)]
        for i, (kind, bad) in enumerate(cases):
            mid = str(rng.choice(eligible))
            k_true, gap_true = float(cat.loc[mid, "k_total"]), float(cat.loc[mid, "band_gap"])
            req = {
                "request_id": f"REQ-eval{lab_id[:2]}{i:03d}",
                "partner_id": "partner-eval",
                "program_id": "prg-eval000000",
                "material_id": mid,
                "formula": cat.loc[mid, "formula"],
            }
            outcome = PlannedOutcome(kind, 0.0, bad_kind=bad, failure_reason="sample cracked" if kind == "failed" else None)
            fname, content = render_raw(lab_id, req, outcome, k_true, gap_true, 10.0, rng)
            (FILES_DIR / fname).write_text(content)
            expect = {"ok": "accept", "failed": "failed_result", "bad_data": "reject"}[kind]
            manifest.append({"file": fname, "lab_id": lab_id, "request": req, "expect": expect, "bad_kind": bad, "k": k_true, "gap_eV": gap_true})
    (FILES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_normalization_eval(rebuild: bool = False, save: bool = True) -> dict:
    mpath = FILES_DIR / "manifest.json"
    manifest = build_messy_set() if rebuild or not mpath.exists() else json.loads(mpath.read_text())
    rows = []
    for m in manifest:
        content = (FILES_DIR / m["file"]).read_text()
        res = adapters.adapt(m["lab_id"], m["file"], content, m["request"], 10.0)
        if not res.accepted:
            got = "reject"
        elif res.result.status == "failed":
            got = "failed_result"
        else:
            got = "accept"
        correct = got == m["expect"]
        value_ok = None
        if correct and got == "accept":
            k = res.result.dielectric_constant.value
            g = res.result.band_gap.value
            value_ok = abs(k - m["k"]) / m["k"] < 5e-3 and abs(g - m["gap_eV"]) < 2e-3
            correct = value_ok
        rows.append({**{k: m[k] for k in ("file", "lab_id", "expect", "bad_kind")}, "got": got, "correct": correct, "reason": res.reason, "conversions": res.conversions})
    by_lab = {}
    for lab in LABS:
        r = [x for x in rows if x["lab_id"] == lab]
        by_lab[lab] = {"files": len(r), "correct": sum(x["correct"] for x in r)}
    summary = {
        "files": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "rate": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "by_lab": by_lab,
        "by_expectation": {e: {"files": sum(r["expect"] == e for r in rows), "correct": sum(r["correct"] for r in rows if r["expect"] == e)} for e in ("accept", "failed_result", "reject")},
        "mistakes": [r for r in rows if not r["correct"]],
        "sample_rejections": [{"file": r["file"], "reason": r["reason"]} for r in rows if r["got"] == "reject"][:8],
    }
    if save:
        out = config.REPORTS_DIR / "platform"
        out.mkdir(parents=True, exist_ok=True)
        (out / "normalization.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return summary


def run_failure_eval(seeds: int = 5, failure_rate: float = 0.30, late_rate: float = 0.40, bad_rate: float = 0.15, save: bool = True) -> dict:
    """Crank every lab's failure, lateness and bad-data rates up; the loop must still finish."""
    from lodestone.labs import mock_labs
    from lodestone.program import ProgramService
    from lodestone.schemas import TargetSpec
    from lodestone.store import Database

    original = {k: v.model_copy() for k, v in mock_labs.LABS.items()}
    rows = []
    try:
        for lab in mock_labs.LABS.values():
            lab.failure_rate, lab.late_rate, lab.bad_data_rate = failure_rate, late_rate, bad_rate
        for seed in range(seeds):
            db = Database(":memory:")
            db.platform().create_partner("stress", "partner-stress")
            svc = ProgramService(db.scope("partner-stress", "eval"))
            t = svc.create_target(TargetSpec(name="stress", k_min=15, gap_min_eV=4, budget=40))
            p = svc.create_program(t["target_id"], seed=100 + seed)
            err = None
            try:
                st = svc.run(p["program_id"])
            except Exception as e:  # the point of the test: this must not happen
                err = f"{type(e).__name__}: {e}"
                st = svc.status(p["program_id"])
            by = st["experiments_by_status"]
            rows.append(
                {
                    "seed": seed,
                    "completed": st["status"] == "complete" and err is None,
                    "error": err,
                    "experiments": st["budget_used"],
                    "returned": by.get("returned", 0),
                    "failed": by.get("failed", 0),
                    "rejected": by.get("rejected", 0),
                    "pending_at_end": by.get("queued", 0) + by.get("in_progress", 0),
                    "hits": len(st["hits"]),
                }
            )
            db.close()
    finally:
        for k, v in original.items():
            mock_labs.LABS[k] = v
    summary = {
        "seeds": seeds,
        "rates": {"failure": failure_rate, "late": late_rate, "bad_data": bad_rate},
        "completed": sum(r["completed"] for r in rows),
        "rows": rows,
    }
    if save:
        out = config.REPORTS_DIR / "platform"
        out.mkdir(parents=True, exist_ok=True)
        (out / "lab_failures.json").write_text(json.dumps(summary, indent=2))
    return summary
