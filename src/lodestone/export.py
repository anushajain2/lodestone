"""FAIR-style JSON export (D5): a campaign's results with full metadata and provenance.

Findable (stable ids, title, keywords), Accessible (plain JSON), Interoperable (explicit units,
schema version, element symbols), Reusable (source citation, method per measurement, provenance
per value back to the HTEM sample and library).
"""

from __future__ import annotations

from datetime import datetime, timezone

from lodestone import __version__
from lodestone.campaign import CampaignService
from lodestone.schemas import PROPERTIES
from lodestone.store import PartnerScope

HTEM_CITATION = (
    "Zakutayev, A. et al. An open experimental database for exploring inorganic materials. "
    "Scientific Data 5, 180053 (2018). Dataset DOI: 10.7799/1407128"
)


def export_campaign(scope: PartnerScope, campaign_id: str) -> dict:
    svc = CampaignService(scope)
    c = scope.get_campaign(campaign_id)
    spec = svc.spec(campaign_id)
    results = scope.list_results(campaign_id)
    records = {r["record_id"]: scope.get_record(r["record_id"]) for r in results}
    doc = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "identifier": campaign_id,
        "name": c["name"],
        "description": f"Closed-loop campaign results: {spec.summary()}. Lab replay of real thin-film measurements.",
        "keywords": ["thin films", "closed-loop optimization", "Bayesian optimization", spec.system],
        "dateCreated": c["created_at"],
        "dateExported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "creator": {"name": scope.partner()["name"]},
        "isBasedOn": HTEM_CITATION,
        "license": "Derived from public HTEM DB data; check the source's terms before redistribution.",
        "softwareVersion": f"lodestone {__version__}",
        "schema": {"record": "lodestone.schemas.ExperimentRecord/1", "units": {p: v["unit"] for p, v in PROPERTIES.items()}},
        "campaign": spec.model_dump(),
        "experiments": [],
    }
    for r in results:
        res = r["result"]
        rec = records[r["record_id"]]
        doc["experiments"].append({
            "result_id": r["result_id"], "round": r["round"], "status": r["status"], "failure_reason": res.get("reason"),
            "record_id": r["record_id"], "composition": rec["composition"], "synthesis": rec["synthesis"],
            "measurements": {p: {"value": v, "unit": PROPERTIES[p]["unit"], "method": rec["measurements"][p]["method"]}
                             for p, v in res.get("values", {}).items() if p in rec["measurements"]},
            "provenance": rec["provenance"],
        })
    scope.log("export", "campaign", campaign_id, detail=f"FAIR JSON, {len(results)} results")
    return doc
