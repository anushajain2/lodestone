import json

import numpy as np

from lodestone.labs import adapters
from lodestone.labs.mock_labs import LABS, PlannedOutcome, render_raw
from lodestone.labs.router import NoCapableLab, route

REQ = {"request_id": "REQ-abc1234567", "partner_id": "p", "program_id": "prg-0123456789", "material_id": "mp-352", "formula": "HfO2"}


def test_router_capability_and_queue():
    assert route("O", {}).lab.lab_id == "helios"  # shortest wait, oxides allowed
    assert route("F", {}).lab.lab_id in {"kestrel", "atlas"}  # Helios cannot make fluorides
    assert route("O", {"helios": 30}).lab.lab_id != "helios"  # long queue → elsewhere
    assert route("F", {"kestrel": 10, "atlas": 0}).lab.lab_id == "atlas"


def test_router_no_capable_lab(monkeypatch):
    import lodestone.labs.mock_labs as ml

    monkeypatch.setattr(ml, "LABS", {"helios": LABS["helios"]})
    import lodestone.labs.router as r

    monkeypatch.setattr(r, "capable_labs", lambda anion, m=None: [l for l in ml.LABS.values() if anion in (l.allowed_anions or [anion])])
    try:
        route("F", {})
        assert False
    except NoCapableLab:
        pass


def _roundtrip(lab, outcome, k=20.0, gap=5.0, seed=0):
    rng = np.random.default_rng(seed)
    fname, content = render_raw(lab, REQ, outcome, k, gap, 3.0, rng)
    return adapters.adapt(lab, fname, content, REQ, 3.0)


def test_each_lab_good_file_converts_to_standard_units():
    for lab in LABS:
        for seed in range(6):  # covers Helios' two firmware formats
            res = _roundtrip(lab, PlannedOutcome("ok", 0), seed=seed)
            assert res.accepted, (lab, res.reason)
            r = res.result
            assert r.dielectric_constant.unit == "dimensionless" and r.band_gap.unit == "eV"
            assert abs(r.dielectric_constant.value - 20.0) < 0.05 and abs(r.band_gap.value - 5.0) < 0.01
            assert r.provenance.lab_id == lab and r.provenance.raw_file


def test_failures_become_failed_results():
    for lab in LABS:
        res = _roundtrip(lab, PlannedOutcome("failed", 0, failure_reason="sample cracked"))
        assert res.accepted and res.result.status == "failed" and "cracked" in res.result.failure_reason


def test_bad_data_rejected_with_reason():
    from lodestone.labs.mock_labs import _BAD_KINDS

    for lab, kinds in _BAD_KINDS.items():
        for kind in kinds:
            res = _roundtrip(lab, PlannedOutcome("bad_data", 0, bad_kind=kind))
            assert not res.accepted, (lab, kind)
            assert res.reason and len(res.reason) > 10


def test_kestrel_units():
    content = json.dumps({"specimen": REQ["request_id"], "compound": "HfO2", "status": "complete",
                          "pellet": {"electrode_area_mm2": 132.73, "thickness_um": 500}, "capacitance_pF": 47.0, "absorption_edge_nm": 248.0})
    res = adapters.adapt("kestrel", "f.json", content, REQ, 1.0)
    assert res.accepted
    assert abs(res.result.band_gap.value - 1239.84 / 248.0) < 1e-3
    assert abs(res.result.dielectric_constant.value - 47e-12 * 500e-6 / (8.8541878128e-12 * 132.73e-6)) < 1e-2


def test_normalization_eval_all_correct():
    from lodestone.evals.platform_eval import run_normalization_eval

    s = run_normalization_eval(save=False)
    assert s["rate"] == 1.0, s["mistakes"]
