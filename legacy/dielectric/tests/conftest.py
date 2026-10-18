import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def seeded(tmp_path_factory):
    """A database with two partners, each with a target, a program and one completed round."""
    from lodestone.program import ProgramService
    from lodestone.schemas import TargetSpec
    from lodestone.store import Database

    path = tmp_path_factory.mktemp("db") / "test.db"
    db = Database(path)
    plat = db.platform()
    plat.create_partner("A", "partner-a", token="tok-a")
    plat.create_partner("B", "partner-b", token="tok-b")
    ids = {}
    for pid, seed in (("partner-a", 1), ("partner-b", 2)):
        scope = db.scope(pid, "test")
        svc = ProgramService(scope)
        t = svc.create_target(TargetSpec(name=f"{pid} target", k_min=15, gap_min_eV=4, budget=20, initial_batch=6, batch_size=4))
        p = svc.create_program(t["target_id"], seed=seed)
        svc.run_round(p["program_id"])
        svc.propose(p["program_id"])  # leaves proposed candidates too
        run_id = scope.save_agent_run(p["program_id"], [{"final_text": "x"}], {"policy": "test"})
        ids[pid] = {
            "target": t["target_id"],
            "program": p["program_id"],
            "candidate": scope.list_candidates(p["program_id"])[0]["candidate_id"],
            "experiment": scope.list_experiments(p["program_id"])[0]["request_id"],
            "raw_file": scope.list_raw_files(p["program_id"])[0]["raw_id"],
            "result": scope.list_results(p["program_id"])[0]["result_id"],
            "agent_run": run_id,
        }
    return db, ids, str(path)
