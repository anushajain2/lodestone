"""Library-level campaigns: one experiment = one deposition run (a whole library of spots)."""

import numpy as np
import pytest
from pydantic import ValidationError

from lodestone.loop.benchmark import public_records
from lodestone.loop.engine import simulate, summarize_runs
from lodestone.loop.problem import Problem
from lodestone.loop.replay import LabReplay
from lodestone.loop.select import library_grid_order
from lodestone.schemas import CampaignSpec

LIB = CampaignSpec(name="t", system="N-Sn-Zn", objectives=[{"property": "conductivity"}], experiment_unit="library",
                   budget=5, initial=2, batch_size=1)
LIB_TC = CampaignSpec(name="t", system="Co-Ni-O-Zn", objectives=[{"property": "conductivity"}], experiment_unit="library",
                      constraints=[{"property": "band_gap", "op": ">=", "value": 3.0}], budget=5, initial=2, batch_size=2, delay_rounds=1)


@pytest.fixture(scope="module")
def zsn():
    return Problem.from_records(public_records("N-Sn-Zn"), LIB)


def runs(t):
    """Run number → library, from a trajectory."""
    out = {}
    for st in t.steps:
        assert out.setdefault(st.n, st.library) == st.library  # every spot of a run comes from one library
    return out


def test_spec_defaults_to_samples_and_validates_library_options():
    assert CampaignSpec(name="t", system="N-Sn-Zn", objectives=[{"property": "conductivity"}]).experiment_unit == "sample"
    with pytest.raises(ValidationError, match="two objectives"):
        CampaignSpec(name="t", system="N-Sn-Zn", objectives=[{"property": "conductivity"}, {"property": "band_gap"}], experiment_unit="library")
    with pytest.raises(ValidationError, match="GP"):
        CampaignSpec(**(LIB.model_dump() | {"surrogate": "rf"}))


@pytest.mark.parametrize("strategy", ["random", "grid", "bayes"])
def test_budget_counts_libraries_and_reveals_every_spot(zsn, strategy):
    t = simulate(zsn, strategy, seed=0)
    r = runs(t)
    libs = zsn.libraries()
    assert sorted(r) == [1, 2, 3, 4, 5]  # 5 experiments = 5 libraries
    assert len(set(r.values())) == 5  # no library is run twice
    assert len(t.steps) == sum(len(libs[lib]) for lib in r.values()) > 5 * 30
    for n, lib in r.items():  # every spot of the chosen library, each once
        assert sorted(st.index for st in t.steps if st.n == n) == sorted(libs[lib].tolist())
    s = summarize_runs(zsn, [t])
    m = s["methods"][t.method]
    assert m["experiments_to_first_top5_median"] <= 6  # counted in runs: at most budget + 1


def test_seed_runs_are_shared_across_strategies(zsn):
    first = [tuple(runs(simulate(zsn, s, seed=3, budget=3)).values())[:2] for s in ("random", "grid", "bayes")]
    assert first[0] == first[1] == first[2]


def test_one_seed_run_then_one_run_per_round(zsn):
    t = simulate(zsn, "random", seed=0, initial=1)
    rounds = {st.n: st.round for st in t.steps}
    assert rounds == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def test_library_replay_returns_each_spots_real_measurements(zsn):
    t = simulate(zsn, "random", seed=1, failure_rate=0.0)
    rep = LabReplay(zsn, 0.0, 1)
    for st in t.steps:
        out = rep.run(st.index)
        assert (st.status, st.kind) == (out.status, out.kind)
        if st.status == "failed":  # with no simulated mishaps, only real gaps in the data fail
            assert st.kind == "missing" and not np.isfinite(zsn.truth["conductivity"][st.index])
        else:
            assert np.isfinite(zsn.truth["conductivity"][st.index])
    assert any(st.status == "failed" for st in simulate(zsn, "random", seed=1, failure_rate=0.3).steps)


def test_budget_is_capped_at_the_number_of_libraries(zsn):
    t = simulate(zsn, "random", seed=0, budget=500)
    assert len(runs(t)) == len(zsn.libraries()) and len(t.steps) == zsn.n


def test_constrained_library_campaign_with_batches_and_delay():
    p = Problem.from_records(public_records("Co-Ni-O-Zn"), LIB_TC)
    t = simulate(p, "bayes", seed=2)
    assert sorted(runs(t)) == [1, 2, 3, 4, 5] and len(set(runs(t).values())) == 5


def test_library_grid_order_is_a_permutation_of_libraries(zsn):
    order = library_grid_order(zsn, 10, seed=0)
    assert sorted(order) == sorted(zsn.libraries())


def test_interactive_service_refuses_library_campaigns(seeded):
    from lodestone.campaign import CampaignError, CampaignService

    db, _, _ = seeded
    with pytest.raises(CampaignError, match="simulation only"):
        CampaignService(db.scope("partner-a")).create(LIB)
