"""Partner programs: the closed loop as a service.

One program = one target + rounds of propose → route → lab → adapt → learn, until the experiment
budget runs out. Everything is read and written through a :class:`~lodestone.store.PartnerScope`,
so the dashboard, the API, the MCP server and the agent all share one partner-safe code path.

Simulated time: each program has its own clock (``sim_day``). Labs return results on their due day
(plus delays for late jobs). ``advance`` moves the clock and ingests whatever has arrived.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from lodestone.data.catalog import load_catalog, material_exists
from lodestone.data.features import nearest_neighbours, reduced_features
from lodestone.explain import why_proposed
from lodestone.labs import adapters
from lodestone.labs.mock_labs import LABS, estimate_costs, measure, plan_outcome, render_raw, synthesis_route
from lodestone.labs.router import NoCapableLab, route
from lodestone.loop.metrics import DEFAULT_REF, figure_of_merit, hypervolume_2d, pareto_mask
from lodestone.loop.strategies import STRATEGIES, Pick, SelectionContext, select
from lodestone.loop.surrogate import Surrogate, to_objectives
from lodestone.schemas import ExperimentRequest, TargetSpec
from lodestone.store import PartnerScope, new_id
from lodestone.targets import filter_pool

MAX_ATTEMPTS = 2  # a material that failed twice is not proposed again


class ProgramError(ValueError):
    pass


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


@dataclass
class ModelState:
    pool: pd.DataFrame
    X: np.ndarray
    tested_idx: np.ndarray
    Y: np.ndarray
    surrogate: Surrogate | None


class ProgramService:
    def __init__(self, scope: PartnerScope):
        self.scope = scope

    # ------------------------------------------------------------------------------- set up

    def create_target(self, spec: TargetSpec) -> dict:
        return self.scope.create_target(spec.name, spec.model_dump())

    def create_program(self, target_id: str, name: str | None = None, strategy: str = "bayes", seed: int = 0) -> dict:
        if strategy not in STRATEGIES:
            raise ProgramError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
        target = self.scope.get_target(target_id)
        spec = TargetSpec(**target["spec"])
        if len(filter_pool(load_catalog(), spec)) < spec.batch_size + 2:
            raise ProgramError("the target's constraints leave too few materials to run a loop; relax them")
        return self.scope.create_program(target_id, name or target["name"], strategy, seed)

    def spec(self, program_id: str) -> TargetSpec:
        program = self.scope.get_program(program_id)
        return TargetSpec(**self.scope.get_target(program["target_id"])["spec"])

    def pool(self, program_id: str) -> pd.DataFrame:
        return filter_pool(load_catalog(), self.spec(program_id))

    # ------------------------------------------------------------------------------- state

    def _measured(self, program_id: str) -> pd.DataFrame:
        rows = self.scope.list_results(program_id, status="ok")
        if not rows:
            return pd.DataFrame(columns=["material_id", "k", "gap"])
        df = pd.DataFrame(rows)[["material_id", "k", "gap"]]
        return df.groupby("material_id", as_index=False).mean()  # repeated measurements → mean

    def _attempts(self, program_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.scope.list_experiments(program_id):
            if e["status"] in ("failed", "rejected"):
                counts[e["material_id"]] = counts.get(e["material_id"], 0) + 1
        return counts

    def _model_state(self, program_id: str, fit: bool = True) -> ModelState:
        pool = self.pool(program_id)
        X = reduced_features(pool.index).values
        measured = self._measured(program_id)
        in_pool = measured[measured["material_id"].isin(pool.index)]
        tested_idx = np.array([pool.index.get_loc(m) for m in in_pool["material_id"]], dtype=int)
        Y = to_objectives(in_pool["k"].values, in_pool["gap"].values) if len(in_pool) else np.zeros((0, 2))
        # Opt-in shared data (W4): anonymized measurements from other partners warm-start the model.
        shared = [r for r in self.scope.shared_pool() if r["material_id"] in pool.index and r["material_id"] not in set(in_pool["material_id"])]
        X_fit, Y_fit = X[tested_idx], Y
        if shared:
            s_idx = np.array([pool.index.get_loc(r["material_id"]) for r in shared])
            X_fit = np.vstack([X_fit, X[s_idx]])
            Y_fit = np.vstack([Y_fit, to_objectives([r["k"] for r in shared], [r["gap"] for r in shared])])
        surrogate = Surrogate(X).fit(X_fit, Y_fit) if fit and len(Y_fit) >= 2 else None
        return ModelState(pool, X, tested_idx, Y, surrogate)

    def budget_used(self, program_id: str) -> int:
        return len(self.scope.list_experiments(program_id))

    # ------------------------------------------------------------------------------ propose

    def propose(self, program_id: str, batch_size: int | None = None, strategy: str | None = None) -> list[dict]:
        program = self.scope.get_program(program_id)
        if program["status"] != "active":
            raise ProgramError(f"program is {program['status']}")
        spec = self.spec(program_id)
        strategy = strategy or program["strategy"]
        remaining = spec.budget - self.budget_used(program_id)
        if remaining <= 0:
            raise ProgramError("experiment budget is used up")

        # Unsubmitted proposals from an earlier call are superseded.
        for c in self.scope.list_candidates(program_id, status="proposed"):
            self.scope.update_candidate(c["candidate_id"], status="skipped")

        torch.manual_seed(int(_rng(program["seed"], "torch", program["round"]).integers(1 << 31)))
        st = self._model_state(program_id)
        pool = st.pool
        busy = {e["material_id"] for e in self.scope.list_experiments(program_id, status=["queued", "in_progress"])}
        tested = set(pool.index[st.tested_idx])
        gave_up = {m for m, n in self._attempts(program_id).items() if n >= MAX_ATTEMPTS}
        unavailable = np.array([i for i, m in enumerate(pool.index) if m in busy or m in tested or m in gave_up], dtype=int)

        initial = len(st.tested_idx) + len(busy) == 0 or (len(st.tested_idx) < 2 and st.surrogate is None)
        rnd = program["round"] + 1
        rng = _rng(program["seed"], "propose", rnd)
        costs = estimate_costs(pool)
        if initial:
            q = min(batch_size or spec.initial_batch, remaining)
            picks = self._space_filling(st.X, unavailable, q, rng)
            strat_used = "initial"
        else:
            q = min(batch_size or spec.batch_size, remaining)
            ctx = SelectionContext(
                X_pool=st.X, tested=st.tested_idx, Y_tested=st.Y, unavailable=unavailable, q=q, rng=rng,
                ref_point=self._ref_point(spec), costs=costs,
                pending=np.array([pool.index.get_loc(m) for m in sorted(busy) if m in pool.index], dtype=int),
            )
            picks = select(strategy, ctx, st.surrogate)
            strat_used = strategy

        # Context for explanations.
        pool_std_median = None
        if st.surrogate is not None:
            _, s_all = st.surrogate.predict(st.X)
            pool_std_median = np.median(s_all, axis=0)
        tested_ids = list(pool.index[st.tested_idx])
        measured = self._measured(program_id).set_index("material_id")

        rows = []
        for rank, p in enumerate(picks, start=1):
            mid = pool.index[p.index]
            row = pool.loc[mid]
            mean, std = p.mean, p.std
            if mean is None and st.surrogate is not None:
                m, s = st.surrogate.predict(st.X[[p.index]])
                mean, std = m[0], s[0]
            nb = None
            if tested_ids:
                (nid, _), = nearest_neighbours(mid, tested_ids, 1)
                nb = (pool.loc[nid, "formula"], float(measured.loc[nid, "k"]), float(measured.loc[nid, "gap"]))
            why = why_proposed(
                formula=row["formula"], anion=row["anion"], e_hull=row["e_hull"], mean=mean, std=std,
                strategy=strat_used, spec=spec, Y_tested=st.Y, pool_std_median=pool_std_median,
                neighbour=nb, extra_notes=p.notes if strat_used != "random" else None, initial=strat_used == "initial",
            )
            pred = None
            if mean is not None:
                pred = {
                    "k_mean": float(10 ** mean[0]), "k_low": float(10 ** (mean[0] - std[0])), "k_high": float(10 ** (mean[0] + std[0])),
                    "gap_mean_eV": float(mean[1]), "gap_std_eV": float(std[1]),
                }
            rows.append(
                {
                    "candidate_id": new_id("cand"), "program_id": program_id, "round": rnd, "rank": rank,
                    "material_id": mid, "formula": row["formula"], "prediction": pred,
                    "acquisition": p.acquisition, "strategy": strat_used, "why": why,
                    "est_cost": float(costs[p.index]), "status": "proposed",
                }
            )
        self.scope.add_candidates(rows)
        if st.surrogate is not None:
            self.scope.add_model_snapshot(
                program_id, rnd, int(len(st.tested_idx)),
                {"features": f"matminer magpie+stoichiometry+valence+density → PCA({st.X.shape[1]})",
                 "objectives": ["log10 k", "band gap (eV)"], "model": "SingleTaskGP (BoTorch), Standardize outputs",
                 "shared_points": int(len(self.scope.shared_pool())), "ref_point": self._ref_point(spec).tolist()},
            )
        return self.scope.list_candidates(program_id, round=rnd, status="proposed")

    @staticmethod
    def _ref_point(spec: TargetSpec) -> np.ndarray:
        """Acquisition reference point: the partner's thresholds where given (so the loop values the
        region they care about), loosened by 25% so the model still gets gradient early on."""
        ref = DEFAULT_REF.copy()
        if spec.k_min is not None:
            ref[0] = max(ref[0], np.log10(spec.k_min * 0.75))
        if spec.gap_min_eV is not None:
            ref[1] = max(0.0, min(spec.gap_min_eV * 0.75, spec.gap_min_eV - 0.5))
        return ref

    @staticmethod
    def _space_filling(X: np.ndarray, unavailable: np.ndarray, q: int, rng: np.random.Generator) -> list[Pick]:
        avail = np.setdiff1d(np.arange(len(X)), unavailable)
        first = int(rng.choice(avail))
        chosen = [first]
        d = np.linalg.norm(X[avail] - X[first], axis=1)
        while len(chosen) < q:
            # maximin with a little randomness: pick among the 5 farthest
            order = np.argsort(-d)[:5]
            nxt = int(avail[rng.choice(order)])
            chosen.append(nxt)
            d = np.minimum(d, np.linalg.norm(X[avail] - X[nxt], axis=1))
            d[np.isin(avail, chosen)] = -1
        return [Pick(index=i, notes=["initial design"]) for i in chosen]

    # ------------------------------------------------------------------------------- submit

    def submit(self, program_id: str, material_ids: list[str] | None = None) -> list[dict]:
        """Turn proposals (or explicitly named materials) into routed experiment requests."""
        program = self.scope.get_program(program_id)
        spec = self.spec(program_id)
        pool = self.pool(program_id)
        proposed = {c["material_id"]: c for c in self.scope.list_candidates(program_id, status="proposed")}
        if material_ids is None:
            material_ids = list(proposed)
        if not material_ids:
            raise ProgramError("nothing to submit: propose a batch first or name material ids")
        unknown = [m for m in material_ids if not material_exists(m)]
        if unknown:
            raise ProgramError(f"unknown material ids (not in the catalog): {', '.join(unknown)}")
        outside = [m for m in material_ids if m not in pool.index]
        if outside:
            raise ProgramError(f"not allowed by this program's target constraints: {', '.join(outside)}")
        busy = {e["material_id"] for e in self.scope.list_experiments(program_id, status=["queued", "in_progress"])}
        dup = [m for m in material_ids if m in busy]
        if dup:
            raise ProgramError(f"already in the lab queue: {', '.join(dup)}")
        remaining = spec.budget - self.budget_used(program_id)
        if len(material_ids) > remaining:
            raise ProgramError(f"only {remaining} experiments left in the budget; tried to submit {len(material_ids)}")

        platform = self.scope._db.platform()
        rnd = program["round"] + 1
        day = program["sim_day"]
        out = []
        for mid in material_ids:
            row = pool.loc[mid]
            queues = platform.lab_queue_lengths()
            try:
                decision = route(row["anion"], queues)
            except NoCapableLab as e:
                raise ProgramError(str(e)) from None
            lab = decision.lab
            request_id = new_id("REQ")
            attempt = sum(e["material_id"] == mid for e in self.scope.list_experiments(program_id)) + 1
            rng = _rng(program["seed"], mid, attempt)  # reproducible lab behaviour for a given seed
            outcome = plan_outcome(lab, rng)
            due = day + lab.turnaround_days + decision.wait_days
            req = ExperimentRequest(
                request_id=request_id, partner_id=self.scope.partner_id, program_id=program_id, round=rnd,
                material_id=mid, formula=row["formula"], synthesis=synthesis_route(row["formula"], row["anion"], lab),
                lab_id=lab.lab_id, submitted_day=day, due_day=round(due, 2),
            )
            self.scope.add_experiment(
                {
                    "request_id": request_id, "program_id": program_id, "round": rnd, "material_id": mid,
                    "formula": row["formula"], "lab_id": lab.lab_id,
                    "request": {**req.model_dump(), "routing_reason": decision.reason, "attempt": attempt},
                    "status": "queued", "submitted_day": day, "due_day": round(due, 2),
                    "actual_day": round(due + outcome.extra_delay_days, 2),
                    "outcome": outcome.kind + (f":{outcome.bad_kind}" if outcome.bad_kind else "") + (f":{outcome.failure_reason}" if outcome.failure_reason else ""),
                }
            )
            if mid in proposed:
                self.scope.update_candidate(proposed[mid]["candidate_id"], status="submitted")
            out.append(self.scope.get_experiment(request_id))
        self.scope.update_program(program_id, round=rnd)
        return out

    # ------------------------------------------------------------------------------ advance

    def advance(self, program_id: str, days: float | None = None) -> dict:
        """Move the program clock forward and ingest everything the labs have returned.

        With no ``days``, jumps to the next day on which at least one result arrives.
        """
        program = self.scope.get_program(program_id)
        pending = self.scope.list_experiments(program_id, status=["queued", "in_progress"])
        day = program["sim_day"]
        if days is None:
            if not pending:
                return {"day": day, "arrived": [], "rejected": [], "failed": [], "still_pending": 0}
            target_day = min(e["actual_day"] for e in pending)
        else:
            target_day = day + days
        arrived, rejected, failed = [], [], []
        for e in sorted(pending, key=lambda e: (e["actual_day"], e["material_id"], e["request"].get("attempt", 1))):
            if e["actual_day"] > target_day + 1e-9:
                if e["status"] == "queued" and e["submitted_day"] + 0.5 <= target_day:
                    self.scope.update_experiment(e["request_id"], status="in_progress")
                continue
            arrived_one = self._lab_returns(program, e)
            if arrived_one["status"] == "ok":
                arrived.append(arrived_one)
            elif arrived_one["status"] == "failed":
                failed.append(arrived_one)
            else:
                rejected.append(arrived_one)
        self.scope.update_program(program_id, sim_day=round(target_day, 2))
        spec = self.spec(program_id)
        if self.budget_used(program_id) >= spec.budget and not self.scope.list_experiments(program_id, status=["queued", "in_progress"]):
            self.scope.update_program(program_id, status="complete")
            self.scope.log("complete", "program", program_id, detail="budget used")
        still = len(self.scope.list_experiments(program_id, status=["queued", "in_progress"]))
        return {"day": round(target_day, 2), "arrived": arrived, "rejected": rejected, "failed": failed, "still_pending": still}

    def _lab_returns(self, program: dict, e: dict) -> dict:
        lab = LABS[e["lab_id"]]
        rng = _rng(program["seed"], e["material_id"], e["request"].get("attempt", 1), "measure")
        kind, _, rest = e["outcome"].partition(":")
        from lodestone.labs.mock_labs import PlannedOutcome

        if kind == "failed":
            outcome = PlannedOutcome("failed", 0.0, failure_reason=rest)
        elif kind == "bad_data":
            outcome = PlannedOutcome("bad_data", 0.0, bad_kind=rest)
        else:
            outcome = PlannedOutcome("ok", 0.0)
        k, gap = measure(e["material_id"], lab, rng)
        filename, content = render_raw(lab.lab_id, e["request"], outcome, k, gap, e["actual_day"], rng)
        res = adapters.adapt(lab.lab_id, filename, content, e["request"], e["actual_day"])
        raw = self.scope.add_raw_file(
            {
                "program_id": program["program_id"], "request_id": e["request_id"], "lab_id": lab.lab_id,
                "filename": filename, "content": content, "received_day": e["actual_day"],
                "adapter_status": "accepted" if res.accepted else "rejected",
                "adapter_reason": res.reason if not res.accepted else (res.conversions or None),
            }
        )
        summary = {"request_id": e["request_id"], "material_id": e["material_id"], "formula": e["formula"], "lab_id": lab.lab_id, "raw_id": raw["raw_id"]}
        if not res.accepted:
            self.scope.update_experiment(e["request_id"], status="rejected")
            self.scope.log("reject", "raw_file", raw["raw_id"], outcome="rejected", detail=res.reason)
            self._mark_candidate(program["program_id"], e["material_id"], "failed")
            return {**summary, "status": "rejected", "reason": res.reason}
        result = res.result.model_dump()
        self.scope.add_result(result)
        if result["status"] == "failed":
            self.scope.update_experiment(e["request_id"], status="failed")
            self._mark_candidate(program["program_id"], e["material_id"], "failed")
            return {**summary, "status": "failed", "reason": result["failure_reason"]}
        self.scope.update_experiment(e["request_id"], status="returned")
        self._mark_candidate(program["program_id"], e["material_id"], "tested")
        return {**summary, "status": "ok", "k": result["dielectric_constant"]["value"], "gap": result["band_gap"]["value"]}

    def _mark_candidate(self, program_id: str, material_id: str, status: str) -> None:
        for c in self.scope.list_candidates(program_id, status="submitted"):
            if c["material_id"] == material_id:
                self.scope.update_candidate(c["candidate_id"], status=status)

    # -------------------------------------------------------------------------------- rounds

    def run_round(self, program_id: str) -> dict:
        """Propose, submit, and wait until this round's batch has come back (late jobs may trail)."""
        cands = self.propose(program_id)
        reqs = self.submit(program_id)
        batch = {r["request_id"] for r in reqs}
        arrived, rejected, failed = [], [], []
        promised = max(r["due_day"] for r in reqs)
        program = self.scope.get_program(program_id)
        # Wait until the promised due date of this batch; anything late arrives in a later round.
        out = self.advance(program_id, days=max(promised - program["sim_day"], 0.0))
        arrived += out["arrived"]
        rejected += out["rejected"]
        failed += out["failed"]
        late = [e for e in self.scope.list_experiments(program_id, status=["queued", "in_progress"]) if e["request_id"] in batch]
        return {
            "round": self.scope.get_program(program_id)["round"],
            "proposed": len(cands),
            "submitted": len(reqs),
            "arrived": arrived,
            "rejected": rejected,
            "failed": failed,
            "late": [e["request_id"] for e in late],
            "day": out["day"],
        }

    def run(self, program_id: str, max_rounds: int = 100, on_round=None) -> dict:
        for _ in range(max_rounds):
            program = self.scope.get_program(program_id)
            spec = self.spec(program_id)
            if program["status"] != "active":
                break
            if self.budget_used(program_id) >= spec.budget:
                # drain late results
                while self.scope.list_experiments(program_id, status=["queued", "in_progress"]):
                    self.advance(program_id)
                break
            info = self.run_round(program_id)
            if on_round:
                on_round(info)
        return self.status(program_id)

    # ------------------------------------------------------------------------------ reading

    def status(self, program_id: str) -> dict:
        program = self.scope.get_program(program_id)
        spec = self.spec(program_id)
        exps = self.scope.list_experiments(program_id)
        measured = self._measured(program_id)
        hits = [r for r in measured.itertuples() if spec.is_hit(r.k, r.gap)]
        Y = to_objectives(measured["k"].values, measured["gap"].values) if len(measured) else np.zeros((0, 2))
        by_status: dict[str, int] = {}
        for e in exps:
            by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        return {
            "program_id": program_id,
            "name": program["name"],
            "status": program["status"],
            "strategy": program["strategy"],
            "round": program["round"],
            "sim_day": program["sim_day"],
            "target": spec.summary(),
            "budget": spec.budget,
            "budget_used": len(exps),
            "experiments_by_status": by_status,
            "materials_measured": int(len(measured)),
            "hits": [{"material_id": h.material_id, "formula": load_catalog().loc[h.material_id, "formula"], "k": round(h.k, 2), "gap_eV": round(h.gap, 2)} for h in hits],
            "hypervolume": round(hypervolume_2d(Y), 4),
        }

    def rounds_progress(self, program_id: str) -> pd.DataFrame:
        """Hypervolume, hits and best k×Eg after each round (from accepted results, by arrival)."""
        spec = self.spec(program_id)
        exps = {e["request_id"]: e for e in self.scope.list_experiments(program_id)}
        rows = []
        for r in self.scope.list_results(program_id, status="ok"):
            e = exps.get(r["request_id"])
            rows.append({"material_id": r["material_id"], "k": r["k"], "gap": r["gap"], "day": r["result"]["provenance"]["received_day"], "round": e["round"] if e else None})
        if not rows:
            return pd.DataFrame(columns=["round", "day", "measured", "hypervolume", "hits", "best_fom"])
        df = pd.DataFrame(rows).sort_values("day")
        out = []
        for day in sorted(df["day"].unique()):
            sub = df[df["day"] <= day].groupby("material_id", as_index=False)[["k", "gap"]].mean()
            Y = to_objectives(sub["k"].values, sub["gap"].values)
            out.append(
                {
                    "day": day,
                    "round": int(df[df["day"] <= day]["round"].max()),
                    "measured": len(sub),
                    "hypervolume": hypervolume_2d(Y),
                    "hits": int(sum(spec.is_hit(k, g) for k, g in zip(sub["k"], sub["gap"]))),
                    "best_fom": float(figure_of_merit(sub["k"], sub["gap"]).max()),
                }
            )
        return pd.DataFrame(out)

    def shortlist(self, program_id: str, n: int = 10) -> list[dict]:
        """Ranked measured materials: hits first, then Pareto-optimal, then by k×Eg."""
        spec = self.spec(program_id)
        measured = self._measured(program_id)
        if measured.empty:
            return []
        cat = load_catalog()
        Y = to_objectives(measured["k"].values, measured["gap"].values)
        front = pareto_mask(Y)
        out = []
        for i, r in enumerate(measured.itertuples()):
            out.append(
                {
                    "material_id": r.material_id,
                    "formula": cat.loc[r.material_id, "formula"],
                    "k": round(float(r.k), 2),
                    "gap_eV": round(float(r.gap), 2),
                    "k_x_gap": round(float(r.k * r.gap), 1),
                    "meets_target": bool(spec.is_hit(r.k, r.gap)),
                    "on_tradeoff_front": bool(front[i]),
                    "e_hull_eV_per_atom": float(cat.loc[r.material_id, "e_hull"]),
                }
            )
        out.sort(key=lambda d: (not d["meets_target"], not d["on_tradeoff_front"], -d["k_x_gap"]))
        for rank, d in enumerate(out, 1):
            d["rank"] = rank
        return out[:n]

    def predictions(self, program_id: str) -> pd.DataFrame:
        """Model predictions for every pool material (for the trade-off plot)."""
        st = self._model_state(program_id)
        df = st.pool[["formula", "anion", "e_hull"]].copy()
        if st.surrogate is None:
            df["k_pred"], df["gap_pred"], df["k_sd_log"], df["gap_sd"] = np.nan, np.nan, np.nan, np.nan
            return df
        m, s = st.surrogate.predict(st.X)
        df["k_pred"], df["gap_pred"] = 10 ** m[:, 0], m[:, 1]
        df["k_sd_log"], df["gap_sd"] = s[:, 0], s[:, 1]
        return df
