"""W2: every read path, tried with the wrong lab's ids, must fail. Paths are discovered automatically."""

import inspect

import pytest

from lodestone.store import READ_PATHS, AccessDenied, NotFound, PartnerScope


def test_every_public_read_method_is_registered():
    public = {n for n, f in inspect.getmembers(PartnerScope, inspect.isfunction) if n.startswith(("get_", "list_"))}
    assert public <= set(READ_PATHS), f"unregistered read paths: {public - set(READ_PATHS)}"


@pytest.mark.parametrize("attacker,victim", [("partner-a", "partner-b"), ("partner-b", "partner-a")])
def test_cross_partner_reads_blocked(seeded, attacker, victim):
    db, ids, _ = seeded
    scope = db.scope(attacker, "attacker")
    for name, meta in READ_PATHS.items():
        fn = getattr(scope, name)
        if meta["by_id"]:
            with pytest.raises(NotFound):
                fn(ids[victim][meta["by_id"]])
        elif meta["by_campaign"]:
            with pytest.raises(NotFound):
                fn(ids[victim]["campaign"])
        else:
            rows = fn()
            assert all(r.get("partner_id", attacker) == attacker for r in rows), name
    # records are dicts without partner_id: check by content
    own = {r["record_id"] for r in scope.list_records()}
    assert ids[victim]["record"] not in own or ids[victim]["record"] in {r["record_id"] for r in db.scope(attacker).list_records()}


def test_own_reads_work(seeded):
    db, ids, _ = seeded
    scope = db.scope("partner-a")
    for kind, oid in ids["partner-a"].items():
        assert getattr(scope, f"get_{kind}")(oid)


def test_cross_partner_writes_blocked(seeded):
    db, ids, _ = seeded
    a = db.scope("partner-a", "attacker")
    with pytest.raises(NotFound):
        a.update_campaign(ids["partner-b"]["campaign"], status="cancelled")
    with pytest.raises(NotFound):
        a.update_experiment(ids["partner-b"]["experiment"], status="returned")
    assert db.scope("partner-b").get_campaign(ids["partner-b"]["campaign"])["status"] == "active"


def test_denied_reads_are_audited(seeded):
    db, ids, _ = seeded
    a = db.scope("partner-a", "attacker")
    with pytest.raises(NotFound):
        a.get_result(ids["partner-b"]["result"])
    assert any(e["outcome"] == "denied" and e["object_id"] == ids["partner-b"]["result"] for e in a.list_audit())


def test_insert_ignores_caller_partner_id(seeded):
    db, _, _ = seeded
    c = db.scope("partner-a").create_campaign({"name": "x", "system": "N-Sn-Zn", "partner_id": "partner-b"})
    assert c["partner_id"] == "partner-a"


def test_unknown_partner_rejected(seeded):
    db, _, _ = seeded
    with pytest.raises(AccessDenied):
        db.scope("partner-zzz")


def test_api_cross_partner(seeded, monkeypatch):
    from fastapi.testclient import TestClient

    import lodestone.api as api

    db, ids, _ = seeded
    monkeypatch.setattr(api, "get_db", lambda: db)
    c = TestClient(api.app)
    a = {"X-API-Key": "tok-a"}
    assert c.get("/campaigns", headers={"X-API-Key": "nope"}).status_code == 401
    assert c.get("/campaigns").status_code == 401
    assert c.get(f"/campaigns/{ids['partner-a']['campaign']}", headers=a).status_code == 200
    cb = ids["partner-b"]["campaign"]
    for path in (f"/campaigns/{cb}", f"/campaigns/{cb}/candidates", f"/campaigns/{cb}/results", f"/campaigns/{cb}/curve",
                 f"/campaigns/{cb}/export", f"/records/{ids['partner-b']['record']}", f"/libraries/{ids['partner-b']['library']}/report"):
        assert c.get(path, headers=a).status_code == 404, path
    assert c.post(f"/campaigns/{cb}/run-round", headers=a).status_code == 404
    assert {x["partner_id"] for x in c.get("/campaigns", headers=a).json()} == {"partner-a"}


def test_toolbox_cannot_reach_other_partner(seeded):
    from lodestone.tools import Toolbox

    db, ids, _ = seeded
    box = Toolbox(db.scope("partner-a"))
    for tool in ("get_status", "run_round", "compare_with_baselines", "suggest_recipes"):
        out = box.call(tool, {"campaign_id": ids["partner-b"]["campaign"]})
        assert "error" in out and "not found" in out["error"], (tool, out)
    assert "error" in box.call("get_record", {"record_id": ids["partner-b"]["record"]})
