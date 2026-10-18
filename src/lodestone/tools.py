"""The tool layer behind the MCP server (A1) with guardrails (A3).

Every tool:
* is bound to one partner's workspace when the server starts, and no tool takes a partner argument;
* validates its inputs with Pydantic and checks every record, library and campaign id;
* returns plain JSON, and ``{"error": ...}`` instead of raising, so an agent can recover;
* is written to the partner's audit log with actor ``agent``.

Numbers come only from tool results: the agent evaluation counts every number in an answer that no
tool returned as an invented number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from lodestone.campaign import RECIPE_WARNING, CampaignError, CampaignService
from lodestone.connectors import htem
from lodestone.goals import parse_goal
from lodestone.onboarding import OnboardingError, import_htem_library, import_htem_system
from lodestone.schemas import PROPERTIES, CampaignSpec
from lodestone.store import AccessDenied, NotFound, PartnerScope

CAMPAIGN_ID = r"^cmp-[0-9a-f]{10}$"
RECORD_ID = r"^rec-[a-z]+-[A-Za-z0-9_\-]+$"
Prop = Literal["conductivity", "band_gap", "visible_transmittance", "sheet_resistance", "thickness"]


class NoInput(BaseModel):
    pass


class LibrariesInput(BaseModel):
    system: str = Field(description="Element system, e.g. 'N-Sn-Zn'")


class ImportInput(BaseModel):
    library_id: int | None = Field(None, description="One HTEM library id")
    system: str | None = Field(None, description="Or a whole system: every library with data")


class FindInput(BaseModel):
    system: str
    min_conductivity: float | None = Field(None, description="S/cm")
    min_band_gap: float | None = Field(None, description="eV")
    max_band_gap: float | None = Field(None, description="eV")
    deposition_temp_c: float | None = Field(None, description="Only samples grown at this temperature")
    library_id: str | None = None
    sort_by: Prop = "conductivity"
    descending: bool = True
    limit: int = Field(10, ge=1, le=100)


class RecordInput(BaseModel):
    record_id: str = Field(pattern=RECORD_ID)


class CreateInput(BaseModel):
    goal: str | None = Field(None, description="Plain English, e.g. 'most conductive zinc tin nitride, 40 experiments'")
    system: str | None = None
    maximize: Prop | None = None
    minimize: Prop | None = None
    band_gap_min: float | None = None
    band_gap_max: float | None = None
    window_low: float | None = Field(None, description="Band-gap window low edge, eV")
    window_high: float | None = None
    budget: int | None = Field(None, ge=5, le=500)
    batch_size: int | None = Field(None, ge=1, le=20)
    strategy: Literal["bayes", "random", "grid"] | None = None
    surrogate: Literal["gp", "rf"] | None = None
    seed: int = 0


class CampaignInput(BaseModel):
    campaign_id: str = Field(pattern=CAMPAIGN_ID)


class ExplainInput(CampaignInput):
    record_id: str = Field(pattern=RECORD_ID)


class CompareInput(CampaignInput):
    seeds: int = Field(3, ge=1, le=10)


class RecipeInput(CampaignInput):
    n: int = Field(3, ge=1, le=10)


@dataclass
class CallLog:
    name: str
    ok: bool
    seconds: float
    error: str | None = None


def _round(v):
    if isinstance(v, float):
        return float(f"{v:.4g}")
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


@dataclass
class Toolbox:
    scope: PartnerScope
    calls: list[CallLog] = field(default_factory=list)

    def __post_init__(self):
        self.scope = self.scope.as_actor("agent")
        self.svc = CampaignService(self.scope)

    def call(self, name: str, args: dict[str, Any]) -> dict:
        spec = TOOLS.get(name)
        t0 = time.time()
        if spec is None:
            out = {"error": f"unknown tool {name!r}"}
        else:
            model, fn, _ = spec
            try:
                out = _round(fn(self, model(**(args or {}))))
            except ValidationError as e:
                out = {"error": "invalid input: " + "; ".join(f"{'.'.join(map(str, err['loc'])) or 'input'}: {err['msg']}" for err in e.errors())}
            except (CampaignError, OnboardingError, ValueError) as e:
                out = {"error": str(e)}
            except NotFound as e:
                out = {"error": f"{e} (in this workspace)"}
            except AccessDenied as e:
                out = {"error": f"access denied: {e}"}
        ok = "error" not in out
        self.calls.append(CallLog(name, ok, time.time() - t0, out.get("error")))
        self.scope.log("tool_call", "tool", name, outcome="ok" if ok else "error", detail=None if ok else out["error"][:300])
        return out

    # ---------------------------------------------------------------------------------- data

    def list_systems(self, a: NoInput) -> dict:
        avail = []
        for s in htem.available_systems():
            libs = htem.load_snapshot(s)
            avail.append({"system": s, "description": htem.DEMO_SYSTEMS.get(s, ""), "libraries": len(libs),
                          "samples": sum(len(lib.get("samples") or []) for lib in libs)})
        return {"available_from_htem": avail, "in_this_workspace": [{"system": k, "records": v} for k, v in self.scope.systems().items()]}

    def list_libraries(self, a: LibrariesInput) -> dict:
        system = "-".join(sorted(a.system.split("-")))
        try:
            libs = htem.load_snapshot(system)
        except FileNotFoundError:
            raise ValueError(f"no HTEM snapshot for {system}; try list_systems")
        imported = {lib["source_library_id"] for lib in self.scope.list_libraries(system)}
        return {"system": system, "libraries": [
            {"library_id": lib["id"], "samples_available": len(lib.get("samples") or []),
             "deposition_temp_c": lib.get("deposition_initial_temp_c"), "date": (lib.get("sample_date") or "")[:10],
             "imported": str(lib["id"]) in imported} for lib in libs]}

    def import_library(self, a: ImportInput) -> dict:
        if (a.library_id is None) == (a.system is None):
            raise ValueError("give exactly one of library_id or system")
        reps = [import_htem_library(self.scope, a.library_id)] if a.library_id is not None else import_htem_system(self.scope, "-".join(sorted(a.system.split("-"))))
        return {"imported_libraries": len(reps), "received": sum(r.received for r in reps), "accepted": sum(r.accepted for r in reps),
                "rejected": sum(r.rejected for r in reps),
                "reasons": {k: sum(r.reasons().get(k, 0) for r in reps) for r in reps for k in r.reasons()}}

    def find_records(self, a: FindInput) -> dict:
        system = "-".join(sorted(a.system.split("-")))
        rows = []
        for r in self.scope.list_records(system):
            m = {p: v["value"] for p, v in r["measurements"].items()}
            t = r["synthesis"].get("deposition_temp_c")
            if a.min_conductivity is not None and not (m.get("conductivity") or -1) >= a.min_conductivity:
                continue
            if a.min_band_gap is not None and not (m.get("band_gap") or -1) >= a.min_band_gap:
                continue
            if a.max_band_gap is not None and not (m.get("band_gap") is not None and m["band_gap"] <= a.max_band_gap):
                continue
            if a.deposition_temp_c is not None and (t is None or abs(t - a.deposition_temp_c) > 0.5):
                continue
            if a.library_id is not None and r["provenance"]["source_library_id"] != str(a.library_id):
                continue
            if a.sort_by not in m:
                continue
            rows.append({"record_id": r["record_id"], "htem_sample_id": r["provenance"]["source_sample_id"],
                         "library_id": r["provenance"]["source_library_id"], "composition": r["composition"],
                         "deposition_temp_c": t, "measurements": m, "url": r["provenance"].get("url")})
        rows.sort(key=lambda x: x["measurements"][a.sort_by], reverse=a.descending)
        return {"matches": len(rows), "returned": min(len(rows), a.limit), "records": rows[: a.limit],
                "units": {p: PROPERTIES[p]["unit"] for p in PROPERTIES}}

    def get_record(self, a: RecordInput) -> dict:
        return self.scope.get_record(a.record_id)

    # ----------------------------------------------------------------------------- campaigns

    def create_campaign(self, a: CreateInput) -> dict:
        notes: list[str] = []
        if a.goal:
            spec, notes = parse_goal(a.goal, list(self.scope.systems()))
            fields = spec.model_dump()
        else:
            if not a.system:
                raise ValueError("give a goal in plain English or a system")
            fields = {"name": "campaign", "system": a.system, "objectives": [], "constraints": []}
        if a.system:
            fields["system"] = a.system
        if a.maximize or a.minimize:
            fields["objectives"] = [{"property": a.maximize or a.minimize, "direction": "max" if a.maximize else "min"}]
            fields["window"] = None
        if a.window_low is not None and a.window_high is not None:
            fields["window"] = {"property": "band_gap", "low": a.window_low, "high": a.window_high}
            fields["objectives"], fields["constraints"] = [], []
        if a.band_gap_min is not None:
            fields["constraints"] = [c for c in fields.get("constraints", []) if not (c["property"] == "band_gap" and c["op"] == ">=")] + [{"property": "band_gap", "op": ">=", "value": a.band_gap_min}]
        if a.band_gap_max is not None:
            fields["constraints"] = [c for c in fields.get("constraints", []) if not (c["property"] == "band_gap" and c["op"] == "<=")] + [{"property": "band_gap", "op": "<=", "value": a.band_gap_max}]
        for k in ("budget", "batch_size", "strategy", "surrogate"):
            if getattr(a, k) is not None:
                fields[k] = getattr(a, k)
        fields["seed"] = a.seed
        if fields.get("name") == "campaign":
            fields["name"] = f"{fields['system']} campaign"
        spec = CampaignSpec(**fields)
        c = self.svc.create(spec)
        return {"campaign_id": c["campaign_id"], "summary": spec.summary(), "reader_notes": notes,
                "pool_size": self.svc.problem(c["campaign_id"]).n}

    def run_round(self, a: CampaignInput) -> dict:
        before = {x["candidate_id"] for x in self.scope.list_candidates(a.campaign_id)}
        out = self.svc.run_round(a.campaign_id)
        batch = [x for x in self.scope.list_candidates(a.campaign_id) if x["candidate_id"] not in before]
        st = self.svc.status(a.campaign_id)
        return {
            "round": out["round"],
            "batch": [{"record_id": x["record_id"], "label": x["label"], "strategy": x["strategy"], "why": x["why"]} for x in batch],
            "results_arrived": out["arrived"], "failed": out["failed"], "still_in_lab": out["still_queued"],
            "budget_used": st["budget_used"], "budget": st["budget"], "campaign_status": st["status"],
            "best_so_far": self._best_brief(st["best"]),
        }

    @staticmethod
    def _best_brief(b: dict | None) -> dict | None:
        if not b:
            return None
        keep = ("record_id", "label", "conductivity", "band_gap", "visible_transmittance", "htem_sample_id", "library", "url", "rank_in_data", "of", "in_top_5pct")
        return {k: b[k] for k in keep if k in b}

    def get_status(self, a: CampaignInput) -> dict:
        st = self.svc.status(a.campaign_id)
        st["best"] = self._best_brief(st["best"])
        return st

    def explain_choice(self, a: ExplainInput) -> dict:
        return self.svc.explain(a.campaign_id, a.record_id)

    def compare_with_baselines(self, a: CompareInput) -> dict:
        out = self.svc.compare_with_baselines(a.campaign_id, seeds=a.seeds)
        out.pop("curves", None)
        return out

    def suggest_recipes(self, a: RecipeInput) -> dict:
        return {"recipes": self.svc.suggest_recipes(a.campaign_id, a.n), "warning": "unverified predictions; not measured. " + RECIPE_WARNING}


TOOLS: dict[str, tuple[type[BaseModel], Callable, str]] = {
    "list_systems": (NoInput, Toolbox.list_systems, "Material systems available from the HTEM lab connector, and what is already imported in this workspace."),
    "list_libraries": (LibrariesInput, Toolbox.list_libraries, "HTEM sample libraries for a system: id, samples, deposition temperature, date, whether imported."),
    "import_library": (ImportInput, Toolbox.import_library, "Import an HTEM library (or a whole system) into this workspace: validate, convert to the standard schema, and report what was accepted or rejected and why."),
    "find_records": (FindInput, Toolbox.find_records, "Query imported records by system with optional filters (conductivity, band gap, deposition temperature, library), sorted. Returns measured values with HTEM sample ids."),
    "get_record": (RecordInput, Toolbox.get_record, "One standard record with composition, synthesis conditions, measurements with units, and full provenance."),
    "create_campaign": (CreateInput, Toolbox.create_campaign, "Create a discovery campaign from a plain-English goal and/or explicit fields (system, objective, band-gap constraint or window, budget). Returns campaign_id."),
    "run_round": (CampaignInput, Toolbox.run_round, "Run one round: the optimizer picks a batch (each with a reason), the lab replay runs it, and results come back. Returns the batch, results, failures and best so far."),
    "get_status": (CampaignInput, Toolbox.get_status, "Campaign status: budget used, experiments by status, best result so far with its HTEM sample id."),
    "explain_choice": (ExplainInput, Toolbox.explain_choice, "Why the optimizer chose a given sample in this campaign, with its prediction. For GP campaigns, low–high is a 68% interval calibrated on held-out data (the raw GP is overconfident)."),
    "compare_with_baselines": (CompareInput, Toolbox.compare_with_baselines, "Replay the same goal and budget with random and grid search on the same data; experiments needed to find a top-5% sample, per method."),
    "suggest_recipes": (RecipeInput, Toolbox.suggest_recipes, "Recipes the model would try next that are NOT in the data (composition, temperature) with predicted values; low–high is a 68% interval calibrated on held-out deposition runs, where the model's predictions were weak. Unverified."),
}


def material_numbers(text: str) -> list[str]:
    import re

    return re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text)
