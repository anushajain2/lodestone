import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def seeded(tmp_path_factory):
    """Two labs, each with imported libraries, a campaign that has run two rounds, and an agent run."""
    from lodestone.campaign import CampaignService
    from lodestone.connectors import htem
    from lodestone.connectors.adapters import HTEMAdapter
    from lodestone.onboarding import import_htem_library
    from lodestone.schemas import CampaignSpec
    from lodestone.store import Database

    path = tmp_path_factory.mktemp("db") / "test.db"
    db = Database(path)
    plat = db.platform()
    plat.create_partner("A", "partner-a", token="tok-a")
    plat.create_partner("B", "partner-b", token="tok-b")
    ids = {}
    for pid, system, seed in (("partner-a", "N-Sn-Zn", 1), ("partner-b", "Co-Ni-O-Zn", 2)):
        scope = db.scope(pid, "test")
        libs = [lib for lib in htem.load_snapshot(system) if HTEMAdapter().adapt_library(lib)[1].accepted >= 30][:3]
        for lib in libs:
            import_htem_library(scope, lib["id"])
        svc = CampaignService(scope)
        c = svc.create(CampaignSpec(name=f"{pid} campaign", system=system, objectives=[{"property": "conductivity"}], budget=20, initial=6, batch_size=4, seed=seed))
        svc.run_round(c["campaign_id"])
        svc.run_round(c["campaign_id"])
        run_id = scope.save_agent_run(c["campaign_id"], [{"final_text": "x"}], {"policy": "test"})
        ids[pid] = {
            "library": scope.list_libraries()[0]["library_ref"],
            "record": scope.list_records()[0]["record_id"],
            "campaign": c["campaign_id"],
            "candidate": scope.list_candidates(c["campaign_id"])[0]["candidate_id"],
            "experiment": scope.list_experiments(c["campaign_id"])[0]["request_id"],
            "result": scope.list_results(c["campaign_id"])[0]["result_id"],
            "agent_run": run_id,
        }
    return db, ids, str(path)
