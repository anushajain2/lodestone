"""W2: every read path, tried with the wrong partner, must fail. Paths are discovered automatically."""

import inspect

import pytest

from lodestone.store import READ_PATHS, NotFound, PartnerScope, AccessDenied


def _attack(scope: PartnerScope, name: str, meta: dict, foreign: dict):
    fn = getattr(scope, name)
    if meta["by_id"]:
        with pytest.raises(NotFound):
            fn(foreign[meta["by_id"]])
        return "id"
    if meta["by_program"]:
        with pytest.raises(NotFound):
            fn(foreign["program"])
        return "program"
    rows = fn()
    assert all(r["partner_id"] == scope.partner_id for r in rows), f"{name} returned another partner's rows"
    return "list"


def test_every_public_read_method_is_registered():
    public = {n for n, f in inspect.getmembers(PartnerScope, inspect.isfunction) if n.startswith(("get_", "list_"))}
    assert public <= set(READ_PATHS), f"unregistered read paths: {public - set(READ_PATHS)}"


@pytest.mark.parametrize("attacker,victim", [("partner-a", "partner-b"), ("partner-b", "partner-a")])
def test_cross_partner_reads_blocked(seeded, attacker, victim):
    db, ids, _ = seeded
    scope = db.scope(attacker, "attacker")
    kinds = [_attack(scope, name, meta, ids[victim]) for name, meta in READ_PATHS.items()]
    assert len(kinds) == len(READ_PATHS) >= 15


def test_own_reads_work(seeded):
    db, ids, _ = seeded
    scope = db.scope("partner-a", "owner")
    for kind, oid in ids["partner-a"].items():
        assert getattr(scope, f"get_{kind}")(oid)


def test_cross_partner_writes_blocked(seeded):
    db, ids, _ = seeded
    scope = db.scope("partner-a", "attacker")
    with pytest.raises(NotFound):
        scope.update_program(ids["partner-b"]["program"], status="cancelled")
    with pytest.raises(NotFound):
        scope.update_experiment(ids["partner-b"]["experiment"], status="returned")
    with pytest.raises(NotFound):
        scope.create_program(ids["partner-b"]["target"], "steal")
    assert db.scope("partner-b").get_program(ids["partner-b"]["program"])["status"] == "active"


def test_denied_reads_are_audited(seeded):
    db, ids, _ = seeded
    scope = db.scope("partner-a", "attacker")
    with pytest.raises(NotFound):
        scope.get_result(ids["partner-b"]["result"])
    denied = [e for e in scope.list_audit() if e["outcome"] == "denied"]
    assert denied and denied[0]["object_id"] == ids["partner-b"]["result"]


def test_insert_ignores_caller_partner_id(seeded):
    db, _, _ = seeded
    scope = db.scope("partner-a")
    t = scope.create_target("x", {"name": "x", "partner_id": "partner-b"})
    assert t["partner_id"] == "partner-a"


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
    assert c.get("/programs", headers={"X-API-Key": "nope"}).status_code == 401
    assert c.get("/programs").status_code == 401
    assert c.get(f"/programs/{ids['partner-a']['program']}", headers=a).status_code == 200
    pb = ids["partner-b"]["program"]
    for path in (f"/programs/{pb}", f"/programs/{pb}/candidates", f"/programs/{pb}/experiments", f"/programs/{pb}/results",
                 f"/programs/{pb}/raw-files", f"/programs/{pb}/shortlist", f"/programs/{pb}/export/optimade", f"/targets/{ids['partner-b']['target']}"):
        assert c.get(path, headers=a).status_code == 404, path
    assert c.post(f"/programs/{pb}/run-round", headers=a).status_code == 404
    listed = c.get("/programs", headers=a).json()
    assert {p["partner_id"] for p in listed} == {"partner-a"}


def test_toolbox_cannot_reach_other_partner(seeded):
    from lodestone.tools import Toolbox

    db, ids, _ = seeded
    box = Toolbox(db.scope("partner-a"))
    for tool in ("get_program_status", "get_shortlist", "propose_batch", "get_results"):
        out = box.call(tool, {"program_id": ids["partner-b"]["program"]})
        assert "error" in out and "not found" in out["error"], (tool, out)


def test_shared_pool_is_opt_in_and_anonymous(seeded):
    db, ids, _ = seeded
    a, b = db.scope("partner-a"), db.scope("partner-b")
    assert a.shared_pool() == []
    a.set_sharing(True)
    a.contribute_shared("mp-352", 20.0, 4.0)
    rows = db.platform().shared_results()
    assert rows and set(rows[0]) == {"material_id", "k", "gap"}
    assert b.shared_pool() == []  # B does not contribute, so B cannot read it
    a.set_sharing(False)
