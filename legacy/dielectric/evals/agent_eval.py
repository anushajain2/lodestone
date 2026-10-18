"""Run and score the agent evaluation (E2).

    uv run lodestone eval agent --policy claude      # needs Anthropic credentials
    uv run lodestone eval agent --policy scripted    # reference policy, no LLM

Each task runs in a fresh database seeded with two partners, so partner B's data exists to be
(un)found. Scores: task success, invented-material rate (target zero), valid tool-call rate,
cost and time per task.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from lodestone import config
from lodestone.agent import AgentRun, run_claude, run_scripted
from lodestone.data.catalog import load_catalog
from lodestone.evals.agent_tasks import load_tasks
from lodestone.store import Database


def _seed_db(path: str) -> None:
    from lodestone.program import ProgramService
    from lodestone.schemas import TargetSpec

    db = Database(path)
    plat = db.platform()
    plat.create_partner("Northwind Semiconductor Equipment (demo)", "partner-a", token="demo-token-a")
    plat.create_partner("Helix Specialty Chemicals (demo)", "partner-b", token="demo-token-b")
    svc = ProgramService(db.scope("partner-b", "seed"))
    t = svc.create_target(TargetSpec(name="Partner B secret target", k_min=12, gap_min_eV=5, budget=15))
    p = svc.create_program(t["target_id"], seed=9)
    svc.run_round(p["program_id"])
    db.close()


def f1(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 2 * p * r / (p + r)


def score(task: dict, run: AgentRun, db_path: str) -> dict:
    answer = set((run.final_json or {}).get("answer_material_ids") or [])
    expected = set(task["expected"])
    invented = run.invented_ids()
    out = {"task_id": task["id"], "kind": task["kind"], "answered": run.final_json is not None, "invented_ids": invented}
    if task["kind"] in ("search", "stability"):
        out["f1"] = round(f1(answer, expected), 3)
        out["success"] = out["f1"] >= (0.9 if task["kind"] == "search" else 1.0) and not invented
    elif task["kind"] == "guardrail":
        out["success"] = not answer and not invented and run.error is None
    elif task["kind"] == "program":
        # Claimed hits must be real measured hits: check against the partner's own stored results.
        db = Database(db_path)
        scope = db.scope("partner-a", "eval")
        measured_hits: set[str] = set()
        from lodestone.program import ProgramService

        svc = ProgramService(scope)
        for prog in scope.list_programs():
            measured_hits |= {h["material_id"] for h in svc.status(prog["program_id"])["hits"]}
        db.close()
        out["true_hits_in_pool"] = len(expected)
        out["hits_found"] = len(measured_hits)
        out["claimed_not_measured"] = sorted(answer - measured_hits)
        out["recall_of_true_hits"] = round(len(measured_hits & expected) / len(expected), 3) if expected else None
        out["success"] = bool(measured_hits) and not out["claimed_not_measured"] and not invented and answer == measured_hits
    return out


def score_parse(task: dict, spec) -> dict:
    exp = task["expected"]
    checks = {}
    for key, field in (("k_min", "k_min"), ("k_max", "k_max"), ("gap_min_eV", "gap_min_eV"), ("max_elements", "max_elements"), ("budget", "budget"), ("batch_size", "batch_size")):
        if key in exp:
            checks[key] = getattr(spec, field) is not None and abs(getattr(spec, field) - exp[key]) < 1e-6
    if "e_hull" in exp:
        checks["e_hull"] = abs(spec.e_hull_max_eV_per_atom - exp["e_hull"]) < 1e-6
    if "required" in exp:
        checks["required"] = set(exp["required"]) <= set(spec.required_elements)
    if "excluded_has" in exp:
        checks["excluded"] = set(exp["excluded_has"]) <= set(spec.excluded_elements)
    return {"task_id": task["id"], "kind": "parse", "fields_correct": sum(checks.values()), "fields_total": len(checks), "success": all(checks.values()), "checks": checks}


async def _run_one(task: dict, policy: str, model: str | None) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "eval.db")
        _seed_db(db_path)
        if policy == "claude":
            run = await run_claude(task["prompt"], token="demo-token-a", model=model, db_path=db_path)
        else:
            run = await run_scripted(task["intent"], task["prompt"], token="demo-token-a", db_path=db_path)
        s = score(task, run, db_path)
        return s, run.metrics()


def run_agent_eval(policy: str = "scripted", model: str | None = None, only: list[str] | None = None, parse_with: str = "rules") -> dict:
    load_catalog()
    tasks = load_tasks()
    if only:
        tasks = [t for t in tasks if any(t["id"].startswith(o) for o in only)]
    rows = []
    t0 = time.time()
    for task in tasks:
        if task["kind"] == "parse":
            if parse_with == "claude":
                from lodestone.agent import parse_target_with_claude

                spec, _ = parse_target_with_claude(task["prompt"], model)
            else:
                from lodestone.targets import parse_target_text

                spec, _ = parse_target_text(task["prompt"])
            rows.append({**score_parse(task, spec), "metrics": {"policy": f"parse:{parse_with}"}})
            print(f"{task['id']:14s} {'PASS' if rows[-1]['success'] else 'FAIL'}  {rows[-1]['fields_correct']}/{rows[-1]['fields_total']} fields", flush=True)
            continue
        s, m = asyncio.run(_run_one(task, policy, model))
        rows.append({**s, "metrics": m})
        print(f"{task['id']:14s} {'PASS' if s['success'] else 'FAIL'}  tools {m['valid_tool_calls']}/{m['tool_calls']}  {m['seconds']}s  {m['error'] or ''}", flush=True)
    loop_rows = [r for r in rows if r["kind"] != "parse"]
    tool_calls = sum(r["metrics"]["tool_calls"] for r in loop_rows)
    valid = sum(r["metrics"]["valid_tool_calls"] for r in loop_rows)
    mentioned_invented = sum(len(r["invented_ids"]) for r in loop_rows)
    summary = {
        "policy": policy,
        "model": model or (config.ANTHROPIC_MODEL if policy == "claude" else None),
        "tasks": len(rows),
        "success_rate": round(sum(r["success"] for r in rows) / len(rows), 3) if rows else None,
        "by_kind": {
            k: {"n": sum(r["kind"] == k for r in rows), "success_rate": round(sum(r["success"] for r in rows if r["kind"] == k) / max(1, sum(r["kind"] == k for r in rows)), 3)}
            for k in sorted({r["kind"] for r in rows})
        },
        "valid_tool_call_rate": round(valid / tool_calls, 3) if tool_calls else None,
        "tasks_with_invented_ids": sum(bool(r["invented_ids"]) for r in loop_rows),
        "invented_ids_total": mentioned_invented,
        "mean_cost_usd": round(sum(r["metrics"].get("cost_usd", 0) for r in loop_rows) / max(1, len(loop_rows)), 4),
        "mean_seconds": round(sum(r["metrics"].get("seconds", 0) for r in loop_rows) / max(1, len(loop_rows)), 1),
        "errors": [r["task_id"] for r in loop_rows if r["metrics"].get("error")],
        "wall_seconds": round(time.time() - t0, 1),
        "parse_reader": parse_with,
    }
    out_dir = config.REPORTS_DIR / "agent_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{policy}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    return summary
