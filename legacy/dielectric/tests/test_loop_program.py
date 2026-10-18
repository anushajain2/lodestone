import numpy as np
import pytest

from lodestone.loop.metrics import hypervolume_2d, pareto_mask


def test_hypervolume_and_front():
    Y = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.5, 1.5]])
    assert list(pareto_mask(Y)) == [True, True, True, False]
    assert hypervolume_2d(Y, np.array([0.0, 0.0])) == pytest.approx(6.0)
    assert hypervolume_2d(np.array([[0.5, 0.5]]), np.array([1.0, 1.0])) == 0.0


def test_strategies_pick_unique_available():
    from lodestone.loop.benchmark import benchmark_pool
    from lodestone.data.features import surrogate_features
    from lodestone.loop.strategies import SelectionContext, select
    from lodestone.loop.surrogate import to_objectives

    pool = benchmark_pool()
    X = surrogate_features(pool.index).values
    rng = np.random.default_rng(0)
    tested = rng.choice(len(X), 12, replace=False)
    Y = to_objectives(pool["k_total"].values[tested], pool["band_gap"].values[tested])
    for strat in ("random", "greedy", "bayes", "bayes_cost"):
        ctx = SelectionContext(X_pool=X, tested=tested, Y_tested=Y, unavailable=tested, q=4, rng=np.random.default_rng(1), costs=np.ones(len(X)))
        idx = [p.index for p in select(strat, ctx)]
        assert len(idx) == 4 == len(set(idx)) and not set(idx) & set(tested), strat


def test_program_runs_to_budget_and_explains(tmp_path):
    from lodestone.program import ProgramError, ProgramService
    from lodestone.schemas import TargetSpec
    from lodestone.store import Database

    db = Database(tmp_path / "p.db")
    db.platform().create_partner("A", "partner-a")
    svc = ProgramService(db.scope("partner-a"))
    t = svc.create_target(TargetSpec(k_min=15, gap_min_eV=4, budget=22, initial_batch=8, batch_size=5))
    p = svc.create_program(t["target_id"], seed=3)
    st = svc.run(p["program_id"])
    assert st["status"] == "complete" and st["budget_used"] == 22
    assert sum(st["experiments_by_status"].values()) == 22
    cands = db.scope("partner-a").list_candidates(p["program_id"])
    assert all(len(c["why"]) > 20 for c in cands)
    assert any(c["prediction"] for c in cands)
    with pytest.raises(ProgramError):
        svc.propose(p["program_id"])
    assert svc.rounds_progress(p["program_id"])["measured"].iloc[-1] == st["materials_measured"]


def test_submit_guardrails(tmp_path):
    from lodestone.program import ProgramError, ProgramService
    from lodestone.schemas import TargetSpec
    from lodestone.store import Database

    db = Database(tmp_path / "g.db")
    db.platform().create_partner("A", "partner-a")
    svc = ProgramService(db.scope("partner-a"))
    t = svc.create_target(TargetSpec(budget=10, initial_batch=5, required_elements=["O"]))
    p = svc.create_program(t["target_id"])
    with pytest.raises(ProgramError, match="unknown material"):
        svc.submit(p["program_id"], ["mp-99999999"])
    with pytest.raises(ProgramError, match="not allowed"):
        svc.submit(p["program_id"], ["mp-23237"])  # BiF3: no oxygen
    svc.submit(p["program_id"], ["mp-352"])
    with pytest.raises(ProgramError, match="already in the lab queue"):
        svc.submit(p["program_id"], ["mp-352"])


def test_loop_survives_lab_failures():
    from lodestone.evals.platform_eval import run_failure_eval

    s = run_failure_eval(seeds=2, save=False)
    assert s["completed"] == 2, s["rows"]
    assert all(r["failed"] + r["rejected"] > 0 for r in s["rows"])


def test_export_optimade(seeded):
    from lodestone.export import export_program

    db, ids, _ = seeded
    doc = export_program(db.scope("partner-a"), ids["partner-a"]["program"])
    assert doc["data"] and doc["data"][0]["type"] == "structures"
    attrs = doc["data"][0]["attributes"]
    assert "_lodestone_dielectric_constant" in attrs and attrs["nelements"] == len(attrs["elements"])
