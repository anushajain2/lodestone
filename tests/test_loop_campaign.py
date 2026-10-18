import numpy as np
import pytest

from lodestone.loop.benchmark import public_records
from lodestone.loop.engine import simulate, summarize_runs
from lodestone.loop.models import Observations, training_data
from lodestone.loop.problem import Problem
from lodestone.loop.select import grid_order
from lodestone.schemas import CampaignSpec

ZSN = CampaignSpec(name="t", system="N-Sn-Zn", objectives=[{"property": "conductivity"}], budget=25, initial=8, batch_size=4)
TC = CampaignSpec(name="t", system="Co-Ni-O-Zn", objectives=[{"property": "conductivity"}], constraints=[{"property": "band_gap", "op": ">=", "value": 3.0}], budget=25, initial=8, batch_size=4)
WIN = CampaignSpec(name="t", system="N-Sn-Zn", window={"property": "band_gap", "low": 1.4, "high": 1.6}, budget=25, initial=8, batch_size=4)


@pytest.fixture(scope="module")
def zsn():
    return Problem.from_records(public_records("N-Sn-Zn"), ZSN)


def test_problem_features_hide_measurements(zsn):
    assert zsn.n > 900
    assert not any("conductivity" in f or "gap" in f for f in zsn.feature_names)
    assert "deposition_temp_c" in zsn.feature_names
    top = zsn.top_set(0.05)
    assert 45 <= len(top) <= 60


def test_scores_respect_constraints_and_windows():
    tc = Problem.from_records(public_records("Co-Ni-O-Zn"), TC)
    sc = tc.true_scores()
    gap = tc.truth["band_gap"]
    assert np.all(~np.isfinite(sc[gap < 3.0]))
    win = Problem.from_records(public_records("N-Sn-Zn"), WIN)
    s = win.true_scores()
    g = win.truth["band_gap"]
    inside = (g >= 1.4) & (g <= 1.6)
    assert np.all(s[inside] == 0) and np.all(s[np.isfinite(g) & ~inside] < 0)


def test_grid_order_is_a_permutation(zsn):
    order = grid_order(zsn, 25, seed=0)
    assert len(order) == len(set(order)) == zsn.n


@pytest.mark.parametrize("strategy,surrogate", [("random", "gp"), ("grid", "gp"), ("bayes", "gp"), ("bayes", "rf")])
def test_simulate_uses_exact_budget_and_records_failures(zsn, strategy, surrogate):
    t = simulate(zsn, strategy, surrogate, seed=0, failure_rate=0.2)
    assert len(t.steps) == 25
    assert len({(s.index, s.n) for s in t.steps}) == 25
    assert any(s.status == "failed" for s in t.steps)
    assert all(s.status in ("ok", "failed") for s in t.steps)


@pytest.mark.parametrize("spec", [TC, WIN])
def test_constrained_and_window_campaigns_run(spec):
    p = Problem.from_records(public_records(spec.system), spec)
    t = simulate(p, "bayes", "gp", seed=1)
    assert len(t.steps) == 25
    s = summarize_runs(p, [t, simulate(p, "random", "gp", seed=1)])
    assert set(s["methods"]) == {"bayes-gp", "random"}


def test_failed_conductivity_is_censored_not_dropped():
    obs = Observations()
    obs.add(0, {"conductivity": 100.0})
    obs.add(1, {"conductivity": 10.0})
    obs.add(2, {}, "missing", "conductivity: no four-point-probe value")
    obs.add(3, {}, "random", "sample lost (simulated)")
    idx, y, n_cens = training_data(obs, ["conductivity"])["conductivity"]
    assert n_cens == 1 and 2 in idx and 3 not in idx
    assert y[list(idx).index(2)] == pytest.approx(0.0)  # one decade below the lowest measured (log10 10 = 1)


def test_campaign_service_end_to_end(seeded):
    from lodestone.campaign import CampaignError, CampaignService
    from lodestone.export import export_campaign

    db, ids, _ = seeded
    scope = db.scope("partner-a")
    svc = CampaignService(scope)
    c = svc.create(CampaignSpec(name="e2e", system="N-Sn-Zn", objectives=[{"property": "conductivity"}], budget=18, initial=6, batch_size=4, delay_rounds=1, failure_rate=0.1, seed=7))
    st = svc.run(c["campaign_id"])
    assert st["status"] == "complete" and st["budget_used"] == 18
    assert sum(st["experiments_by_status"].values()) == 18 and "queued" not in st["experiments_by_status"]
    b = st["best"]
    assert b and b["htem_sample_id"] and b["url"] and 1 <= b["rank_in_data"] <= b["of"]
    cands = scope.list_candidates(c["campaign_id"])
    assert all(len(x["why"]) > 20 for x in cands)
    assert any(x["prediction"] for x in cands)
    curve = svc.curve(c["campaign_id"])
    assert len(curve) == 18
    with pytest.raises(CampaignError):
        svc.propose(c["campaign_id"])
    cmp = svc.compare_with_baselines(c["campaign_id"], seeds=2)
    assert {"random", "grid", "bayes-gp"} <= set(cmp["methods"]) and cmp["experiments_saved_vs"] is not None
    recipes = svc.suggest_recipes(c["campaign_id"], 3)
    assert len(recipes) == 3 and all("unverified" in r["status"] and "deposition run" in r["warning"] for r in recipes)
    assert all(v["interval"] == "68% interval, calibrated on held-out deposition runs" for r in recipes for v in r["predicted"].values())
    doc = export_campaign(scope, c["campaign_id"])
    assert len(doc["experiments"]) == len(scope.list_results(c["campaign_id"])) and doc["schema"]["units"]["conductivity"] == "S/cm"
    assert all(e["provenance"]["source_sample_id"] for e in doc["experiments"])


def test_campaign_guardrails(seeded):
    from lodestone.campaign import CampaignError, CampaignService

    db, ids, _ = seeded
    svc = CampaignService(db.scope("partner-a"))
    with pytest.raises(CampaignError, match="no Co-Ni-O-Zn data"):
        svc.create(CampaignSpec(name="x", system="Co-Ni-O-Zn", objectives=[{"property": "conductivity"}]))
    with pytest.raises(CampaignError, match="constraints"):
        svc.create(CampaignSpec(name="x", system="N-Sn-Zn", objectives=[{"property": "conductivity"}], constraints=[{"property": "band_gap", "op": ">=", "value": 7.5}]))
    c = svc.create(CampaignSpec(name="g", system="N-Sn-Zn", objectives=[{"property": "conductivity"}], budget=10, initial=5, batch_size=5))
    with pytest.raises(CampaignError, match="not samples"):
        svc.submit(c["campaign_id"], ["rec-htem-1"])
    svc.propose(c["campaign_id"])
    subs = svc.submit(c["campaign_id"])
    with pytest.raises(CampaignError, match="already"):
        svc.submit(c["campaign_id"], [subs[0]["record_id"]])


def test_std_inflation_keeps_the_mean_and_scales_the_std(zsn):
    from lodestone.loop.models import GP_OPTIONS, Surrogate, gp_options
    from lodestone.loop.problem import transform

    y = transform("conductivity", zsn.truth["conductivity"])
    idx = np.flatnonzero(np.isfinite(y))[::20][:30]
    data = {"conductivity": (idx, y[idx], 0)}
    base = Surrogate(zsn.X).fit(data).predict(zsn.X[:100])["conductivity"]
    with gp_options(std_inflation={"conductivity": 1.7}):
        wide = Surrogate(zsn.X).fit(data).predict(zsn.X[:100])["conductivity"]
    assert np.allclose(base[0], wide[0], atol=1e-9)
    assert np.allclose(wide[1] / base[1], 1.7, rtol=1e-6)
    assert GP_OPTIONS["std_inflation"] is None  # the context manager restores the defaults


def _tc_obs(tc, n=12):
    ok = np.flatnonzero(np.isfinite(tc.truth["conductivity"]) & np.isfinite(tc.truth["band_gap"]))
    obs = Observations()
    for i in np.random.default_rng(0).choice(ok, n, replace=False):
        obs.add(int(i), {p: float(tc.truth[p][i]) for p in ("conductivity", "band_gap")})
    return obs


def test_display_inflation_widens_what_people_see_but_not_the_picks(monkeypatch):
    from lodestone.loop import select
    from lodestone.loop.calibration import display_inflation
    from lodestone.loop.held_out_libraries import recipe_inflation
    from lodestone.loop.models import GP_OPTIONS

    tc = Problem.from_records(public_records(TC.system), TC)
    infl = display_inflation(TC.system)
    assert 1.2 < infl["conductivity"] < 2.0 and 1.5 < infl["band_gap"] < 2.6
    obs = _tc_obs(tc)

    def picks():
        out, _ = select.select_bayes(tc, obs, obs.tested(), [], 4, np.random.default_rng(5), "gp", seed=5)
        return out

    shown = picks()
    monkeypatch.setattr(select, "display_inflation", lambda system: {})
    monkeypatch.setattr(select, "recipe_inflation", lambda system: {})
    raw = picks()
    assert [p.index for p in shown] == [p.index for p in raw]
    assert [p.acquisition for p in shown] == [p.acquisition for p in raw]
    new = recipe_inflation(TC.system)
    seen = {tc.library[i] for i in obs.tested()}
    for a, b in zip(shown, raw):
        assert a.new_run == (tc.library[a.index] not in seen) and not b.new_run
        assert a.std_inflation == (new if a.new_run else infl) and b.std_inflation == {}
        infl_a = a.std_inflation
        for prop in ("conductivity", "band_gap"):
            assert a.prediction[prop][0] == b.prediction[prop][0]
            assert a.prediction[prop][1] == pytest.approx(b.prediction[prop][1] * infl_a[prop])
        assert abs(a.p_feasible - 0.5) <= abs(b.p_feasible - 0.5) + 1e-12  # wider σ → less certain about the gap constraint
    assert GP_OPTIONS["std_inflation"] is None and not GP_OPTIONS["calibrate"]


@pytest.mark.parametrize("surrogate", ["gp", "rf"])
def test_seeded_simulate_picks_the_same_samples_with_and_without_display_inflation(monkeypatch, surrogate):
    from lodestone.loop import select

    tc = Problem.from_records(public_records(TC.system), TC)
    after = [s.index for s in simulate(tc, "bayes", surrogate, seed=3).steps]
    monkeypatch.setattr(select, "display_inflation", lambda system: {})
    monkeypatch.setattr(select, "recipe_inflation", lambda system: {})
    before = [s.index for s in simulate(tc, "bayes", surrogate, seed=3).steps]
    assert after == before


def test_explanations_label_the_calibrated_interval():
    from lodestone.campaign import CampaignService
    from lodestone.loop import select

    tc = Problem.from_records(public_records(TC.system), TC)
    pk = select.select_bayes(tc, _tc_obs(tc), set(), [], 1, np.random.default_rng(5), "gp", seed=5)[0][0]
    why = CampaignService.why(None, tc, _tc_obs(tc), pk)
    pj = CampaignService._prediction_json(tc, pk)
    label = "calibrated on held-out deposition runs" if pk.new_run else "calibrated on held-out data"
    assert label in why and pj["conductivity"]["interval"] == f"68% interval, {label}" and pj["band_gap"]["std_inflation"] > 1
    assert pj.get("new_run", False) == pk.new_run


def test_picks_from_a_deposition_run_with_no_data_get_the_wider_new_run_range():
    from lodestone.campaign import CampaignService
    from lodestone.loop import select
    from lodestone.loop.calibration import display_inflation
    from lodestone.loop.held_out_libraries import recipe_inflation

    tc = Problem.from_records(public_records(TC.system), TC)
    obs = _tc_obs(tc, 6)  # 6 samples cannot cover the 20 libraries, so unseen-run picks exist
    seen = {tc.library[i] for i in obs.tested()}
    seen_runs = {i for i in range(tc.n) if tc.library[i] in seen}  # make every sample from a seen run unavailable
    picks, _ = select.select_bayes(tc, obs, seen_runs, [], 3, np.random.default_rng(1), "gp", seed=1)
    new, disp = recipe_inflation(TC.system), display_inflation(TC.system)
    assert picks and all(pk.new_run and pk.std_inflation == {p: new[p] for p in pk.std_inflation} for pk in picks)
    assert all(new[p] >= disp[p] for p in ("conductivity", "band_gap"))
    assert "deposition run" in CampaignService.why(None, tc, obs, picks[0])


def test_recipe_inflation_is_never_narrower_than_the_display_factor():
    from lodestone.loop.calibration import display_inflation
    from lodestone.loop.held_out_libraries import recipe_inflation

    for system in ("N-Sn-Zn", "Co-Ni-O-Zn", "In-O-Sn-Zn", "a system with no checks"):
        rec, disp = recipe_inflation(system), display_inflation(system)
        assert set(disp) <= set(rec) and all(rec[p] >= disp[p] for p in disp)


def test_leave_one_library_out_holds_out_whole_libraries():
    from lodestone.loop import held_out_libraries as h

    res = h.check(sizes=(), targets=(("izto-conductivity", "conductivity"),), verbose=False)
    (row,) = res["results"]
    assert row["n_train"] == "all" and row["libraries"] == len(row["per_library"]) >= 2
    assert row["library"]["n_predictions"] == row["random"]["n_predictions"] == sum(p["n_test"] for p in row["per_library"])
    for arm in ("library", "random"):
        assert 0 <= row[arm]["cov68"] <= row[arm]["cov95"] <= 1
    assert all(p["best_gap"] >= 0 and 0 < p["pick_percentile"] <= 1 for p in row["per_library"])
