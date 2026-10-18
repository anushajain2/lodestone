"""Partner campaigns: the closed loop as a service, on the partner's own imported lab data.

One campaign = one target on one material system + rounds of propose → run (lab replay) → learn,
until the budget is used. Everything is read and written through a
:class:`~lodestone.store.PartnerScope`, so the dashboard, API, MCP server and agent share one
partner-safe code path. The selection logic is the same code the benchmark uses
(:mod:`lodestone.loop.engine`).
"""

from __future__ import annotations

import numpy as np

from lodestone.loop.calibration import INTERVAL_LABEL
from lodestone.loop.engine import initial_picks, next_batch, simulate, summarize_runs, best_so_far, MAX_RANDOM_RETRIES
from lodestone.loop.held_out_libraries import RECIPE_INTERVAL_LABEL, recipe_inflation
from lodestone.loop.models import Observations, Surrogate, training_data
from lodestone.loop.problem import Problem, fmt_value, transform, untransform
from lodestone.loop.replay import LabReplay
from lodestone.loop.select import Pick, grid_order
from lodestone.schemas import PROPERTIES, CampaignSpec
from lodestone.store import PartnerScope, new_id


# Why suggested recipes carry a warning: reports/EVALUATION.md, 'Can it predict a deposition run it has never seen?'
# (reports/calibration/held_out_libraries.json, as of 2026-10-18).
RECIPE_WARNING = (
    "A new recipe means a new deposition run. Trained on 30-60 samples and tested on whole deposition runs it had not seen, the model "
    "explained little of the variation (R² from -0.8 to 0.5, median about 0, across the demo systems), so treat the predicted value as a rough guess."
)


class CampaignError(ValueError):
    pass


def fmt(prop: str, v: float) -> str:
    unit = PROPERTIES[prop]["unit"]
    if prop == "conductivity":
        return f"{fmt_value(v)} {unit}"
    if unit == "fraction":
        return f"{100 * v:.0f}%"
    return f"{v:.2f} {unit}"


def _interval_note(std_inflation: dict[str, float], prop: str, label: str = INTERVAL_LABEL) -> dict:
    """How to read low–high: ±1σ of the GP, widened by the held-out calibration factor when there is one."""
    if prop in std_inflation:
        return {"interval": label, "std_inflation": std_inflation[prop]}
    return {"interval": "±1σ of the model, not calibrated"}


class CampaignService:
    def __init__(self, scope: PartnerScope):
        self.scope = scope
        self._problems: dict[tuple, Problem] = {}

    # ------------------------------------------------------------------------------ set up

    def create(self, spec: CampaignSpec) -> dict:
        if spec.experiment_unit == "library":
            raise CampaignError("library-level campaigns (one experiment = one deposition run) run in simulation only in this prototype: "
                                "`uv run lodestone bench --level library`. The interactive loop proposes samples.")
        records = self.scope.list_records(system=spec.system)
        if not records:
            have = ", ".join(self.scope.systems()) or "nothing yet"
            raise CampaignError(f"no {spec.system} data in this workspace (imported: {have}). Import a library first.")
        for p in spec.properties_needed():
            n = sum(p in r["measurements"] for r in records)
            if n < 10:
                raise CampaignError(f"only {n} {spec.system} samples have {PROPERTIES[p]['label'].lower()} measured; need at least 10")
        prob = Problem.from_records(records, spec)
        if spec.kind != "pareto" and not np.isfinite(prob.true_scores()).any():
            raise CampaignError("no sample in this data meets the constraints at all; the campaign could never succeed. Relax the target.")
        if spec.budget > prob.n:
            raise CampaignError(f"budget {spec.budget} exceeds the {prob.n} samples available")
        return self.scope.create_campaign(spec.model_dump())

    def spec(self, campaign_id: str) -> CampaignSpec:
        return CampaignSpec(**self.scope.get_campaign(campaign_id)["spec"])

    def problem(self, campaign_id: str) -> Problem:
        c = self.scope.get_campaign(campaign_id)
        records = self.scope.list_records(system=c["system"])
        key = (campaign_id, len(records))
        if key not in self._problems:
            self._problems[key] = Problem.from_records(records, CampaignSpec(**c["spec"]))
        return self._problems[key]

    def _index(self, prob: Problem) -> dict[str, int]:
        return {rid: i for i, rid in enumerate(prob.ids)}

    def observations(self, campaign_id: str) -> tuple[Observations, dict[int, int]]:
        prob = self.problem(campaign_id)
        idx = self._index(prob)
        obs = Observations()
        attempts: dict[int, int] = {}
        for r in self.scope.list_results(campaign_id):
            i = idx.get(r["record_id"])
            if i is None:
                continue
            res = r["result"]
            attempts[i] = max(attempts.get(i, 0), res.get("attempt", 1))
            obs.add(i, res.get("values", {}), res.get("kind"), res.get("reason"))
        return obs, attempts

    def budget_used(self, campaign_id: str) -> int:
        return len(self.scope.list_experiments(campaign_id))

    # ------------------------------------------------------------------------------ the loop

    def _unavailable(self, campaign_id: str, prob: Problem, obs: Observations, attempts: dict[int, int]) -> tuple[set[int], list[int]]:
        idx = self._index(prob)
        pending = [idx[e["record_id"]] for e in self.scope.list_experiments(campaign_id, status="queued") if e["record_id"] in idx]
        gave_up = {i for i, kind, _ in obs.failures if kind == "missing" or (kind == "random" and attempts.get(i, 0) > MAX_RANDOM_RETRIES)}
        retryable = {i for i, kind, _ in obs.failures if kind == "random" and i not in gave_up and not any(i in v for v in obs.values.values())}
        return (obs.tested() | set(pending) | gave_up) - retryable, pending

    def propose(self, campaign_id: str, batch_size: int | None = None) -> list[dict]:
        c = self.scope.get_campaign(campaign_id)
        if c["status"] != "active":
            raise CampaignError(f"campaign is {c['status']}")
        spec, prob = self.spec(campaign_id), self.problem(campaign_id)
        remaining = spec.budget - self.budget_used(campaign_id)
        if remaining <= 0:
            raise CampaignError("experiment budget is used up")
        for old in self.scope.list_candidates(campaign_id, status="proposed"):
            self.scope.update_candidate(old["candidate_id"], status="superseded")
        obs, attempts = self.observations(campaign_id)
        unavailable, pending = self._unavailable(campaign_id, prob, obs, attempts)
        rnd = c["round"] + 1
        first = self.budget_used(campaign_id) == 0
        if first:
            q = min(batch_size or spec.initial, remaining)
            picks, meta = initial_picks(prob, unavailable, q, spec.seed), {}
            for p in picks:
                p.strategy = "initial"
        else:
            q = min(batch_size or spec.batch_size, remaining)
            grid = grid_order(prob, spec.budget, spec.seed) if spec.strategy == "grid" else None
            picks, meta = next_batch(prob, obs, unavailable, pending, q, spec.strategy, spec.surrogate, spec.seed, rnd, grid)
        if meta.get("surrogate"):
            self.scope.add_model_snapshot(campaign_id, rnd, meta)
        rows = []
        for rank, pk in enumerate(picks, 1):
            rows.append({
                "candidate_id": new_id("cand"), "campaign_id": campaign_id, "round": rnd, "rank": rank,
                "record_id": prob.ids[pk.index], "label": prob.labels[pk.index],
                "prediction": self._prediction_json(prob, pk), "acquisition": pk.acquisition,
                "strategy": pk.strategy, "why": self.why(prob, obs, pk), "status": "proposed",
            })
        self.scope.add_candidates(rows)
        return self.scope.list_candidates(campaign_id, round=rnd, status="proposed")

    @staticmethod
    def _prediction_json(prob: Problem, pk: Pick) -> dict | None:
        if not pk.prediction:
            return None
        out = {}
        for p, (m, s) in pk.prediction.items():
            lo, mid, hi = untransform(p, np.array([m - s, m, m + s]))
            out[p] = {"mean": float(mid), "low": float(lo), "high": float(hi), "unit": PROPERTIES[p]["unit"], **_interval_note(pk.std_inflation, p, RECIPE_INTERVAL_LABEL if pk.new_run else INTERVAL_LABEL)}
        if pk.p_feasible is not None:
            out["p_feasible"] = pk.p_feasible
        if pk.p_success is not None:
            out["p_success"] = pk.p_success
        if pk.new_run:
            out["new_run"] = True  # no measurements from this sample's deposition run yet: widened for a new run
        return out

    def submit(self, campaign_id: str, record_ids: list[str] | None = None) -> list[dict]:
        c = self.scope.get_campaign(campaign_id)
        spec, prob = self.spec(campaign_id), self.problem(campaign_id)
        idx = self._index(prob)
        proposed = {x["record_id"]: x for x in self.scope.list_candidates(campaign_id, status="proposed")}
        if record_ids is None:
            record_ids = list(proposed)
        if not record_ids:
            raise CampaignError("nothing to submit: propose a batch first or name record ids")
        unknown = [r for r in record_ids if r not in idx]
        if unknown:
            raise CampaignError(f"not samples of this campaign's system in this workspace: {unknown}")
        obs, attempts = self.observations(campaign_id)
        unavailable, _ = self._unavailable(campaign_id, prob, obs, attempts)
        taken = [r for r in record_ids if idx[r] in unavailable]
        if taken:
            raise CampaignError(f"already tested, in the queue, or failed for good: {taken}")
        remaining = spec.budget - self.budget_used(campaign_id)
        if len(record_ids) > remaining:
            raise CampaignError(f"only {remaining} experiments left in the budget")
        rnd = c["round"] + 1
        out = []
        for rid in record_ids:
            req = new_id("REQ")
            attempt = attempts.get(idx[rid], 0) + 1
            rec = prob.records[idx[rid]]
            self.scope.add_experiment({
                "request_id": req, "campaign_id": campaign_id, "round": rnd, "record_id": rid, "label": prob.labels[idx[rid]],
                "status": "queued", "due_round": rnd + spec.delay_rounds,
                "plan": {"attempt": attempt, "how_to_make": rec["synthesis"], "target_composition": rec["composition"],
                         "measure": spec.properties_needed(), "source_library_id": rec["provenance"]["source_library_id"],
                         "position": rec["provenance"]["position"]},
            })
            if rid in proposed:
                self.scope.update_candidate(proposed[rid]["candidate_id"], status="submitted")
            out.append(self.scope.get_experiment(req))
        self.scope.update_campaign(campaign_id, round=rnd)
        self.scope.log("submit", "experiments", campaign_id, detail=f"{len(out)} in round {rnd}")
        return out

    def advance(self, campaign_id: str) -> dict:
        """Collect everything the lab has finished by now (results due this round)."""
        c = self.scope.get_campaign(campaign_id)
        spec, prob = self.spec(campaign_id), self.problem(campaign_id)
        idx = self._index(prob)
        replay = LabReplay(prob, spec.failure_rate, spec.seed)
        arrived, failed = [], []
        queued = self.scope.list_experiments(campaign_id, status="queued")
        rnd = c["round"]
        if queued and all(e["due_round"] > rnd for e in queued):
            rnd = min(e["due_round"] for e in queued)  # nothing due yet: wait for the lab
            self.scope.update_campaign(campaign_id, round=rnd)
        for e in queued:
            if e["due_round"] > rnd:
                continue
            i = idx[e["record_id"]]
            out = replay.run(i, e["plan"]["attempt"])
            rec = prob.records[i]
            prov = rec["provenance"]
            self.scope.add_result({
                "result_id": new_id("res"), "campaign_id": campaign_id, "request_id": e["request_id"], "record_id": e["record_id"],
                "status": out.status, "round": rnd,
                "result": {
                    "status": out.status, "kind": out.kind, "reason": out.reason, "attempt": e["plan"]["attempt"],
                    "values": {p: v for p, v in out.values.items() if v is not None},
                    "units": {p: PROPERTIES[p]["unit"] for p, v in out.values.items() if v is not None},
                    "label": prob.labels[i],
                    "provenance": {**prov, "measurement_methods": {p: rec["measurements"][p]["method"] for p in out.values if out.values[p] is not None and p in rec["measurements"]}},
                },
            })
            self.scope.update_experiment(e["request_id"], status="returned" if out.status == "ok" else "failed")
            for cand in self.scope.list_candidates(campaign_id, status="submitted"):
                if cand["record_id"] == e["record_id"]:
                    self.scope.update_candidate(cand["candidate_id"], status="tested" if out.status == "ok" else "failed")
            (arrived if out.status == "ok" else failed).append({"record_id": e["record_id"], "label": prob.labels[i], "reason": out.reason, **{p: v for p, v in out.values.items() if v is not None and p in spec.properties_needed()}})
        if self.budget_used(campaign_id) >= spec.budget and not self.scope.list_experiments(campaign_id, status="queued"):
            self.scope.update_campaign(campaign_id, status="complete")
            self.scope.log("complete", "campaign", campaign_id, detail="budget used")
        return {"round": rnd, "arrived": arrived, "failed": failed, "still_queued": len(self.scope.list_experiments(campaign_id, status="queued"))}

    def run_round(self, campaign_id: str) -> dict:
        c = self.scope.get_campaign(campaign_id)
        spec = self.spec(campaign_id)
        proposed = []
        if self.budget_used(campaign_id) < spec.budget and c["status"] == "active":
            proposed = self.propose(campaign_id)
            self.submit(campaign_id)
        out = self.advance(campaign_id)
        return {**out, "proposed": len(proposed)}

    def run(self, campaign_id: str, max_rounds: int = 500, on_round=None) -> dict:
        for _ in range(max_rounds):
            if self.scope.get_campaign(campaign_id)["status"] != "active":
                break
            info = self.run_round(campaign_id)
            if on_round:
                on_round(info)
        return self.status(campaign_id)

    # ------------------------------------------------------------------------------ reading

    def results_table(self, campaign_id: str) -> list[dict]:
        rows = []
        for n, r in enumerate(self.scope.list_results(campaign_id), 1):
            res = r["result"]
            rows.append({"n": n, "round": r["round"], "record_id": r["record_id"], "label": res["label"], "status": r["status"],
                         "reason": res.get("reason"), **res.get("values", {}),
                         "htem_sample_id": res["provenance"]["source_sample_id"], "library": res["provenance"]["source_library_id"],
                         "url": res["provenance"].get("url")})
        return rows

    def best(self, campaign_id: str) -> dict | None:
        prob = self.problem(campaign_id)
        if prob.spec.kind == "pareto":
            return None
        idx = self._index(prob)
        best, best_row = -np.inf, None
        for row in self.results_table(campaign_id):
            if row["status"] != "ok":
                continue
            vals = {p: np.array([row.get(p, np.nan)], float) for p in prob.truth}
            sc = prob.score_from_values(vals)[0]
            if sc > best:
                best, best_row = sc, row
        if best_row is None:
            return None
        all_sc = prob.true_scores()
        rank = int((all_sc > best).sum()) + 1  # infeasible or unmeasured samples count as worse
        return {**best_row, "score": float(best), "text": prob.natural_best(best),
                "rank_in_data": rank, "of": prob.n, "top_percent": round(100.0 * rank / prob.n, 2),
                "in_top_5pct": bool(idx[best_row["record_id"]] in set(prob.top_set(0.05)))}

    def curve(self, campaign_id: str) -> list[dict]:
        """Best true value found against experiments used (in submission order)."""
        prob = self.problem(campaign_id)
        if prob.spec.kind == "pareto":
            return []
        idx = self._index(prob)
        res_by_req = {r["request_id"]: r for r in self.scope.list_results(campaign_id)}
        best, out = -np.inf, []
        sc = prob.true_scores()
        for n, e in enumerate(self.scope.list_experiments(campaign_id), 1):
            r = res_by_req.get(e["request_id"])
            if r and r["status"] == "ok":
                best = max(best, sc[idx[e["record_id"]]])
            out.append({"experiments": n, "best_score": float(best) if np.isfinite(best) else None, "pending": r is None})
        return out

    def status(self, campaign_id: str) -> dict:
        c = self.scope.get_campaign(campaign_id)
        spec = self.spec(campaign_id)
        exps = self.scope.list_experiments(campaign_id)
        by: dict[str, int] = {}
        for e in exps:
            by[e["status"]] = by.get(e["status"], 0) + 1
        b = self.best(campaign_id)
        return {
            "campaign_id": campaign_id, "name": c["name"], "status": c["status"], "round": c["round"],
            "target": spec.summary(), "kind": spec.kind, "budget": spec.budget, "budget_used": len(exps),
            "experiments_by_status": by, "best": b,
        }

    def why(self, prob: Problem, obs: Observations, pk: Pick) -> str:
        spec = prob.spec
        rec = prob.records[pk.index]
        syn = rec["synthesis"]
        made = f"made by {syn['method']}" + (f" at {syn['deposition_temp_c']:.0f} °C" if syn.get("deposition_temp_c") is not None else "")
        if pk.strategy == "initial":
            return f"Seed experiment: a random sample to give the model a first look across the library ({made})."
        if pk.strategy == "random":
            return f"Random baseline pick ({made})."
        if pk.strategy == "grid":
            return f"Next point in a planned sweep over composition and temperature ({made})."
        if not pk.prediction:
            return "Exploring: not enough measurements yet to fit a model."
        parts = []
        main = spec.window.property if spec.window else spec.objectives[0].property
        m, s = pk.prediction[main]
        lo, mid, hi = untransform(main, np.array([m - s, m, m + s]))
        label = RECIPE_INTERVAL_LABEL if pk.new_run else INTERVAL_LABEL
        rng_text = f"{label}: {fmt(main, lo)} to {fmt(main, hi)}" if main in pk.std_inflation else f"likely {fmt(main, lo)} to {fmt(main, hi)}"
        parts.append(f"Predicted {PROPERTIES[main]['label'].lower()} ≈ {fmt(main, mid)} ({rng_text})")
        if spec.window:
            inside = spec.window.low <= mid <= spec.window.high
            parts.append("inside the target window" if inside else f"near the {spec.window.low:g}–{spec.window.high:g} window")
        else:
            best = None
            vals = obs.values.get(main, {})
            feas = [i for i in vals if all(c.ok(obs.values.get(c.property, {}).get(i)) for c in spec.constraints)]
            if feas:
                best = max((vals[i] for i in feas), key=lambda v: v if spec.objectives[0].direction == "max" else -v)
                parts.append(f"best so far is {fmt(main, best)}")
        if pk.new_run:
            parts.append("no measurements from this deposition run yet, so the range is wider: the model was much less accurate on runs it had not seen")
        if pk.p_feasible is not None:
            parts.append(f"{100 * pk.p_feasible:.0f}% chance it meets {', '.join(c.text() for c in spec.constraints)}")
        rel_unc = s / (np.std(transform(main, np.array(list(obs.values.get(main, {0: 1}).values())))) + 1e-9)
        if rel_unc > 0.8:
            parts.append("the model is unsure here, so the result will teach it a lot")
        if pk.p_success is not None and pk.p_success < 0.9:
            parts.append(f"{100 * pk.p_success:.0f}% chance the measurement works (nearby films failed before)")
        tested = sorted(obs.tested())
        if tested:
            d = np.linalg.norm((prob.X[tested] - prob.X[pk.index]) / (prob.X.std(0) + 1e-9), axis=1)
            j = tested[int(np.argmin(d))]
            v = obs.values.get(main, {}).get(j)
            parts.append(f"closest tested sample: {prob.labels[j]}" + (f" measured {fmt(main, v)}" if v is not None else " (measurement failed)"))
        text = "; ".join(parts)
        return text[0].upper() + text[1:] + f". {made[0].upper() + made[1:]}."

    def explain(self, campaign_id: str, record_id: str) -> dict:
        cands = [c for c in self.scope.list_candidates(campaign_id) if c["record_id"] == record_id]
        if not cands:
            raise CampaignError(f"{record_id} was never proposed in this campaign")
        c = cands[-1]
        return {"record_id": record_id, "label": c["label"], "round": c["round"], "strategy": c["strategy"], "why": c["why"], "prediction": c["prediction"], "status": c["status"]}

    def compare_with_baselines(self, campaign_id: str, seeds: int = 5, methods=(("random", "gp"), ("grid", "gp"), ("bayes", "gp"))) -> dict:
        """Replay the same target with the same budget under other strategies, on the same data."""
        prob = self.problem(campaign_id)
        spec = prob.spec
        if spec.kind == "pareto":
            raise CampaignError("baseline comparison uses a single score; not available for two-objective campaigns")
        trajs, curves = [], {}
        for strat, sur in methods:
            for s in range(seeds):
                t = simulate(prob, strat, sur, seed=spec.seed + s)
                trajs.append(t)
                curves.setdefault(t.method, []).append(best_so_far(prob, t))
        summary = summarize_runs(prob, trajs)
        mean_curves = {}
        for m, cs in curves.items():
            arr = np.array(cs)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            mean_curves[m] = [None if not np.isfinite(v) else float(v) for v in np.nanmean(arr, axis=0)]
        meth = summary["methods"]
        saved = None
        if "bayes-gp" in meth:
            ref = {k: v["experiments_to_top_median"] for k, v in meth.items() if k != "bayes-gp"}
            saved = {k: v - meth["bayes-gp"]["experiments_to_top_median"] for k, v in ref.items()}
        summary.update({
            "curves": mean_curves, "seeds": seeds, "budget": spec.budget,
            "experiments_saved_vs": saved,
            "money_saved_vs": None if saved is None else {k: v * spec.cost_per_experiment for k, v in saved.items()},
            "note": "Medians count 'not found within budget' as budget+1.",
        })
        self.scope.log("compare", "campaign", campaign_id, detail=f"{seeds} seeds")
        return summary

    def suggest_recipes(self, campaign_id: str, n: int = 5) -> list[dict]:
        """New recipes the model would try next: NOT in the data, so unverified until someone makes them."""
        prob = self.problem(campaign_id)
        spec = prob.spec
        if spec.kind == "pareto":
            raise CampaignError("recipe suggestions are for single-score campaigns")
        obs, _ = self.observations(campaign_id)
        props = spec.properties_needed()
        data = training_data(obs, props)
        if any(len(data[p][0]) < 3 for p in props):
            raise CampaignError("run a few rounds first: the model needs measurements")
        sur = Surrogate(prob.X, "gp", seed=spec.seed).fit(data)
        ok = [i for i in obs.tested() if all(i in obs.values.get(p, {}) for p in props)]
        sc = prob.score_from_values({p: np.array([obs.values.get(p, {}).get(i, np.nan) for i in ok]) for p in prob.truth}) if ok else np.array([])
        top = [ok[j] for j in np.argsort(-sc)[:5]] if len(sc) else list(range(5))
        rng = np.random.default_rng(spec.seed + 7)
        lo, hi = prob.X.min(0), prob.X.max(0)
        cands = []
        for _ in range(3000):
            a, b = rng.choice(top, 2)
            w = rng.random()
            x = w * prob.X[a] + (1 - w) * prob.X[b] + rng.normal(0, 0.08, prob.X.shape[1]) * (hi - lo)
            cands.append(np.clip(x, lo, hi))
        C = np.array(cands)
        pred = sur.predict(C)
        # A recipe that was never made is a new deposition run: widen by the held-out-library factor, not the
        # random-split one (loop/held_out_libraries.py). Display only; it does not change the ranking.
        infl = {p: f for p, f in recipe_inflation(spec.system).items() if p in props}
        pred = {p: (m, s * infl.get(p, 1.0)) for p, (m, s) in pred.items()}
        vals = {p: untransform(p, pred[p][0]) for p in props}
        full = {p: vals.get(p, np.full(len(C), np.nan)) for p in prob.truth}
        score = prob.score_from_values(full)
        order = np.argsort(-np.where(np.isfinite(score), score, -1e9))
        out, seen = [], set()
        for j in order:
            recipe = self._recipe(prob, C[j])
            key = recipe["text"]
            if key in seen:
                continue
            seen.add(key)
            out.append({**recipe, "predicted": {p: {"mean": float(untransform(p, np.array([pred[p][0][j]]))[0]),
                                                    "low": float(untransform(p, np.array([pred[p][0][j] - pred[p][1][j]]))[0]),
                                                    "high": float(untransform(p, np.array([pred[p][0][j] + pred[p][1][j]]))[0]),
                                                    "unit": PROPERTIES[p]["unit"], **_interval_note(infl, p, RECIPE_INTERVAL_LABEL)} for p in props},
                        "status": "unverified: not in the data; needs a real experiment", "warning": RECIPE_WARNING})
            if len(out) == n:
                break
        return out

    @staticmethod
    def _recipe(prob: Problem, x: np.ndarray) -> dict:
        f = dict(zip(prob.feature_names, x))
        metals = sorted({el for r in prob.records for el in r["composition"]})
        fr = {el: max(0.0, f.get(f"x_{el}", np.nan)) for el in metals if f"x_{el}" in f}
        missing = [el for el in metals if el not in fr]
        if len(missing) == 1:
            fr[missing[0]] = max(0.0, 1 - sum(fr.values()))
        s = sum(fr.values()) or 1.0
        fr = {k: round(float(v / s), 2) for k, v in fr.items()}
        parts = [" ".join(f"{k}{v:.2f}" for k, v in fr.items())]
        if "deposition_temp_c" in f:
            parts.append(f"{round(f['deposition_temp_c'] / 10) * 10:.0f} °C")
        if "growth_pressure_mtorr" in f:
            parts.append(f"{f['growth_pressure_mtorr']:.1f} mTorr")
        if "thickness_um" in f:
            parts.append(f"{1000 * f['thickness_um']:.0f} nm thick")
        return {"composition": fr, "conditions": {k: float(v) for k, v in f.items() if not k.startswith("x_")}, "text": ", ".join(parts)}
