"""Run and score the agent test set (E2).

    uv run lodestone eval agent --policy claude      # needs Anthropic credentials
    uv run lodestone eval agent --policy scripted    # reference policy, no LLM

Each task runs against a fresh copy of a seeded database: partner A has N-Sn-Zn and In-O-Sn-Zn
imported; partner B has Co-Ni-O-Zn and a campaign, so another partner's data exists to be (not) found.
Scores: task success, valid tool calls, invented ids and invented numbers (target zero), cost and time.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

from lodestone import config
from lodestone.agent import AgentRun, run_claude, run_scripted
from lodestone.evals.agent_tasks import load_tasks
from lodestone.store import Database

OUT = config.REPORTS_DIR / "agent_eval"


def seed_template(path: str) -> None:
    from lodestone.campaign import CampaignService
    from lodestone.onboarding import import_htem_system
    from lodestone.schemas import CampaignSpec

    db = Database(path)
    plat = db.platform()
    plat.create_partner("Northwind Thin Films (eval)", "partner-a", token="eval-token-a")
    plat.create_partner("Helix Materials Lab (eval)", "partner-b", token="eval-token-b")
    a = db.scope("partner-a", "eval-seed")
    for s in ("N-Sn-Zn", "In-O-Sn-Zn"):
        import_htem_system(a, s)
    b = db.scope("partner-b", "eval-seed")
    import_htem_system(b, "Co-Ni-O-Zn")
    svc = CampaignService(b)
    c = svc.create(CampaignSpec(name="Helix secret campaign", system="Co-Ni-O-Zn", objectives=[{"property": "conductivity"}], budget=15))
    svc.run(c["campaign_id"])
    db.close()


def _close(a, b, tol=0.01) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1e-9, abs(b))


def score(task: dict, run: AgentRun, db_path: str) -> dict:
    fj = run.final_json or {}
    ids = [str(x) for x in (fj.get("answer_ids") or [])]
    val = fj.get("answer_value")
    exp = task["expected"]
    out = {"task_id": task["id"], "kind": task["kind"], "answered": bool(run.final_json), "invented_ids": run.invented_ids(), "invented_numbers": run.invented_numbers()}
    clean = not out["invented_ids"] and not out["invented_numbers"]
    k = task["kind"]
    if k == "lookup":
        id_ok = exp["record_id"] in ids or any(exp["record_id"].endswith(i) for i in ids)
        out["success"] = id_ok and _close(val, exp["value"]) and clean
    elif k == "count":
        out["success"] = _close(val, exp["value"], 0) and clean
    elif k == "provenance":
        out["success"] = str(val) == str(exp["value"]) and clean
    elif k == "campaign":
        db = Database(db_path)
        from lodestone.campaign import CampaignService

        scope = db.scope("partner-a", "eval")
        svc = CampaignService(scope)
        bests = []
        for c in scope.list_campaigns():
            if c["status"] == "complete":
                b = svc.best(c["campaign_id"])
                if b:
                    bests.append(b)
        db.close()
        field = exp["answer_field"]
        ok = any((b["record_id"] in ids or b["htem_sample_id"] in ids) and _close(val, b.get(field)) for b in bests)
        out["campaigns_completed"] = len(bests)
        out["success"] = ok and clean
    elif k == "guardrail":
        out["success"] = not ids and val is None and clean and run.error is None
    return out


def score_parse(task: dict, spec) -> dict:
    exp = task["expected"]
    checks = {"system": spec.system == exp["system"]}
    if "maximize" in exp:
        checks["maximize"] = bool(spec.objectives) and spec.objectives[0].property == exp["maximize"] and spec.objectives[0].direction == "max"
    if "window" in exp:
        checks["window"] = spec.window is not None and abs(spec.window.low - exp["window"][0]) < 1e-9 and abs(spec.window.high - exp["window"][1]) < 1e-9
    if "gap_min" in exp:
        checks["gap_min"] = any(c.property == "band_gap" and c.op == ">=" and abs(c.value - exp["gap_min"]) < 1e-9 for c in spec.constraints)
    if "gap_max" in exp:
        checks["gap_max"] = any(c.property == "band_gap" and c.op == "<=" and abs(c.value - exp["gap_max"]) < 1e-9 for c in spec.constraints)
    for f in ("budget", "batch_size"):
        if f in exp:
            checks[f] = getattr(spec, f) == exp[f]
    return {"task_id": task["id"], "kind": "parse", "success": all(checks.values()), "fields_correct": sum(checks.values()), "fields_total": len(checks), "checks": checks}


async def _run_one(task: dict, policy: str, model: str | None, template: str) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "eval.db")
        shutil.copy(template, db_path)
        if policy == "claude":
            run = await run_claude(task["prompt"], token="eval-token-a", model=model, db_path=db_path)
        else:
            run = await run_scripted(task["intent"], task["prompt"], token="eval-token-a", db_path=db_path)
        return score(task, run, db_path), run.metrics()


def run_agent_eval(policy: str = "scripted", model: str | None = None, only: list[str] | None = None, parse_with: str = "rules") -> dict:
    tasks = load_tasks()
    if only:
        tasks = [t for t in tasks if any(t["id"].startswith(o) for o in only)]
    rows = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        template = str(Path(tmp) / "template.db")
        seed_template(template)
        for task in tasks:
            if task["kind"] == "parse":
                if parse_with == "claude":
                    from lodestone.agent import parse_goal_with_claude

                    spec, _ = parse_goal_with_claude(task["prompt"], model)
                else:
                    from lodestone.goals import parse_goal

                    spec, _ = parse_goal(task["prompt"])
                rows.append({**score_parse(task, spec), "invented_ids": [], "invented_numbers": [], "metrics": {"policy": f"parse:{parse_with}"}})
            else:
                s, m = asyncio.run(_run_one(task, policy, model, template))
                rows.append({**s, "metrics": m})
            r = rows[-1]
            print(f"{task['id']:15s} {'PASS' if r['success'] else 'FAIL'}  {r['metrics'].get('valid_tool_calls', '')}/{r['metrics'].get('tool_calls', '')}  {r['metrics'].get('error') or ''}", flush=True)
    loop = [r for r in rows if r["kind"] != "parse"]
    calls = sum(r["metrics"]["tool_calls"] for r in loop)
    valid = sum(r["metrics"]["valid_tool_calls"] for r in loop)
    guard_calls = sum(r["metrics"]["tool_calls"] for r in loop if r["kind"] == "guardrail")
    guard_valid = sum(r["metrics"]["valid_tool_calls"] for r in loop if r["kind"] == "guardrail")
    summary = {
        "policy": policy, "model": model or (config.ANTHROPIC_MODEL if policy == "claude" else None),
        "tasks": len(rows), "success_rate": round(sum(r["success"] for r in rows) / len(rows), 3),
        "by_kind": {k: {"n": sum(r["kind"] == k for r in rows), "success_rate": round(sum(r["success"] for r in rows if r["kind"] == k) / max(1, sum(r["kind"] == k for r in rows)), 3)}
                    for k in sorted({r["kind"] for r in rows})},
        "valid_tool_call_rate": round(valid / calls, 3) if calls else None,
        "valid_tool_call_rate_excluding_guardrails": round((valid - guard_valid) / (calls - guard_calls), 3) if calls - guard_calls else None,
        "tasks_with_invented_numbers": sum(bool(r["invented_numbers"]) for r in loop),
        "tasks_with_invented_ids": sum(bool(r["invented_ids"]) for r in loop),
        "mean_cost_usd": round(sum(r["metrics"].get("cost_usd", 0) for r in loop) / max(1, len(loop)), 4),
        "mean_seconds": round(sum(r["metrics"].get("seconds", 0) for r in loop) / max(1, len(loop)), 1),
        "errors": [r["task_id"] for r in loop if r["metrics"].get("error")],
        "wall_seconds": round(time.time() - t0, 1), "parse_reader": parse_with,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{policy}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    return summary
