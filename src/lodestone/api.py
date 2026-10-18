"""REST API (FastAPI). Every route is partner-scoped by the ``X-API-Key`` header.

    uv run lodestone api      → http://127.0.0.1:8000/docs

Simple partner scoping, not production auth: one static token per partner (hashed at rest).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lodestone.campaign import CampaignError, CampaignService
from lodestone.export import export_campaign
from lodestone.goals import parse_goal
from lodestone.onboarding import OnboardingError, import_htem_library, import_partner_csv
from lodestone.schemas import CampaignSpec
from lodestone.store import AccessDenied, Database, NotFound, PartnerScope

app = FastAPI(
    title="Lodestone API", version="0.2.0",
    description="Partner-scoped API: onboard lab data, run closed-loop campaigns on it, read results with provenance.",
)


@lru_cache(maxsize=1)
def get_db() -> Database:
    return Database()


def scope(x_api_key: Annotated[str | None, Header()] = None) -> PartnerScope:
    if not x_api_key:
        raise HTTPException(401, "missing X-API-Key header")
    db = get_db()
    try:
        return db.scope(db.platform().partner_for_token(x_api_key), actor="api")
    except AccessDenied:
        raise HTTPException(401, "invalid API key") from None


Scope = Annotated[PartnerScope, Depends(scope)]


@app.exception_handler(NotFound)
async def _nf(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(CampaignError)
async def _ce(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(OnboardingError)
async def _oe(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/me")
def me(s: Scope):
    return {**s.partner(), "systems": s.systems()}


@app.get("/libraries")
def libraries(s: Scope, system: str | None = None):
    return [{k: v for k, v in lib.items() if k != "report"} for lib in s.list_libraries(system)]


@app.get("/libraries/{library_ref}/report")
def library_report(library_ref: str, s: Scope):
    return s.get_library(library_ref)["report"]


@app.post("/libraries/htem/{library_id}")
def import_htem(library_id: int, s: Scope):
    return import_htem_library(s, library_id)


class CSVBody(BaseModel):
    system: str
    filename: str = "export.csv"
    content: str


@app.post("/libraries/csv")
def import_csv(body: CSVBody, s: Scope):
    """How a second lab sends data: post its CSV export; every row is validated, converted, or rejected with a reason."""
    return import_partner_csv(s, body.content, body.system, body.filename)


@app.get("/records/{record_id}")
def record(record_id: str, s: Scope):
    return s.get_record(record_id)


class GoalBody(BaseModel):
    goal: str | None = None
    spec: CampaignSpec | None = None


@app.post("/campaigns/parse")
def parse(body: GoalBody, s: Scope):
    if not body.goal:
        raise HTTPException(422, "goal required")
    spec, notes = parse_goal(body.goal, list(s.systems()))
    return {"spec": spec, "summary": spec.summary(), "notes": notes}


@app.post("/campaigns")
def create_campaign(body: GoalBody, s: Scope):
    spec = body.spec or (parse_goal(body.goal, list(s.systems()))[0] if body.goal else None)
    if spec is None:
        raise HTTPException(422, "give goal or spec")
    return CampaignService(s).create(spec)


@app.get("/campaigns")
def campaigns(s: Scope):
    return s.list_campaigns()


@app.get("/campaigns/{campaign_id}")
def campaign_status(campaign_id: str, s: Scope):
    return CampaignService(s).status(campaign_id)


@app.post("/campaigns/{campaign_id}/run-round")
def run_round(campaign_id: str, s: Scope):
    return CampaignService(s).run_round(campaign_id)


@app.get("/campaigns/{campaign_id}/candidates")
def candidates(campaign_id: str, s: Scope):
    return s.list_candidates(campaign_id)


@app.get("/campaigns/{campaign_id}/results")
def results(campaign_id: str, s: Scope):
    return CampaignService(s).results_table(campaign_id)


@app.get("/campaigns/{campaign_id}/curve")
def curve(campaign_id: str, s: Scope):
    return CampaignService(s).curve(campaign_id)


@app.post("/campaigns/{campaign_id}/compare")
def compare(campaign_id: str, s: Scope, seeds: int = 3):
    return CampaignService(s).compare_with_baselines(campaign_id, seeds)


@app.get("/campaigns/{campaign_id}/recipes")
def recipes(campaign_id: str, s: Scope, n: int = 5):
    return CampaignService(s).suggest_recipes(campaign_id, n)


@app.get("/campaigns/{campaign_id}/export")
def export(campaign_id: str, s: Scope):
    return export_campaign(s, campaign_id)


@app.get("/audit")
def audit(s: Scope, limit: int = 200):
    return s.list_audit(limit)
