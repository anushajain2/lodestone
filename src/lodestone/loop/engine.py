"""Simulated campaigns and their scores (E1).

``simulate`` runs a whole campaign in memory (no store) with any strategy. The benchmark and the
"compare with baselines" tool both use it, so the numbers a partner sees for their campaign and the
numbers in the evaluation report come from the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lodestone.loop.models import Observations
from lodestone.loop.problem import Problem
from lodestone.loop.replay import LabReplay, seeded_rng
from lodestone.loop.select import Pick, grid_order, select_bayes, select_grid, select_random

MAX_RANDOM_RETRIES = 1


@dataclass
class Step:
    n: int  # experiment number (1-based, submission order)
    index: int
    round: int
    status: str = "pending"
    kind: str | None = None
    reason: str | None = None


@dataclass
class Trajectory:
    method: str
    seed: int
    steps: list[Step] = field(default_factory=list)


def method_name(strategy: str, surrogate: str) -> str:
    return f"bayes-{surrogate}" if strategy == "bayes" else strategy


def initial_picks(problem: Problem, unavailable: set[int], q: int, seed: int) -> list[Pick]:
    """The shared seed batch: identical for every strategy with the same seed."""
    return select_random(problem, unavailable, q, np.random.default_rng(20_000 + seed))


def next_batch(problem: Problem, obs: Observations, unavailable: set[int], pending: list[int], q: int,
               strategy: str, surrogate: str, seed: int, round_no: int, grid: list[int] | None = None) -> tuple[list[Pick], dict]:
    rng = seeded_rng(seed, "select", round_no)
    if strategy == "random":
        return select_random(problem, unavailable, q, rng), {}
    if strategy == "grid":
        return select_grid(problem, unavailable, q, grid or grid_order(problem, problem.spec.budget, seed)), {}
    return select_bayes(problem, obs, unavailable, pending, q, rng, surrogate, seed=seed + round_no)


def simulate(problem: Problem, strategy: str, surrogate: str = "gp", seed: int = 0, budget: int | None = None,
             batch: int | None = None, initial: int | None = None, failure_rate: float | None = None,
             delay: int | None = None) -> Trajectory:
    spec = problem.spec
    budget = budget or spec.budget
    batch = batch or spec.batch_size
    initial = initial or spec.initial
    failure_rate = spec.failure_rate if failure_rate is None else failure_rate
    delay = spec.delay_rounds if delay is None else delay
    replay = LabReplay(problem, failure_rate, seed)
    grid = grid_order(problem, budget, seed) if strategy == "grid" else None
    traj = Trajectory(method_name(strategy, surrogate), seed)
    obs = Observations()
    attempts: dict[int, int] = {}
    in_flight: list[tuple[Step, int]] = []  # (step, due round)
    gave_up: set[int] = set()
    used, rnd = 0, 0
    while used < budget or in_flight:
        rnd += 1
        if used < budget:
            unavailable = obs.tested() | {s.index for s, _ in in_flight} | gave_up
            # a random mishap can be retried; a real missing measurement can't
            unavailable -= {i for i, kind, _ in obs.failures if kind == "random" and attempts.get(i, 0) <= MAX_RANDOM_RETRIES and i not in gave_up and not any(i in v for v in obs.values.values())}
            q = min(initial if used == 0 else batch, budget - used)
            if used == 0:
                picks = initial_picks(problem, unavailable, q, seed)
            else:
                picks, _ = next_batch(problem, obs, unavailable, [s.index for s, _ in in_flight], q, strategy, surrogate, seed, rnd, grid)
            for pk in picks:
                used += 1
                attempts[pk.index] = attempts.get(pk.index, 0) + 1
                st = Step(n=used, index=pk.index, round=rnd)
                traj.steps.append(st)
                in_flight.append((st, rnd + delay))
        arrived = [(s, d) for s, d in in_flight if d <= rnd]
        in_flight = [(s, d) for s, d in in_flight if d > rnd]
        for st, _ in arrived:
            out = replay.run(st.index, attempts[st.index])
            st.status, st.kind, st.reason = out.status, out.kind, out.reason
            obs.add(st.index, out.values, out.kind, out.reason)
            if out.kind == "missing" or (out.kind == "random" and attempts[st.index] > MAX_RANDOM_RETRIES):
                gave_up.add(st.index)
        if used >= budget and not in_flight:
            break
    return traj


# ------------------------------------------------------------------------------------- metrics


def best_so_far(problem: Problem, traj: Trajectory) -> np.ndarray:
    """True score of the best successful experiment after each experiment (submission order)."""
    sc = problem.true_scores()
    best, out = -np.inf, []
    for st in traj.steps:
        if st.status == "ok":
            best = max(best, sc[st.index])
        out.append(best)
    return np.array(out)


def experiments_to_top(problem: Problem, traj: Trajectory, top: np.ndarray, k: int = 1) -> int | None:
    """Experiments used until k distinct top samples have been measured successfully (None if never)."""
    top_set = set(int(i) for i in top)
    found: set[int] = set()
    for st in traj.steps:
        if st.status == "ok" and st.index in top_set:
            found.add(st.index)
            if len(found) >= k:
                return st.n
    return None


def top_found(problem: Problem, traj: Trajectory, top: np.ndarray) -> int:
    top_set = set(int(i) for i in top)
    return len({st.index for st in traj.steps if st.status == "ok" and st.index in top_set})


HEADLINE_K = 5  # "find the top 5%" = measure 5 distinct samples from the top-5% set


def summarize_runs(problem: Problem, trajs: list[Trajectory], top_fraction: float = 0.05) -> dict:
    """Per method: experiments to find 5 top-5% samples (headline), to the first top-5% and the first
    top-1% sample, how many top-5% samples were found, the best value reached, and failures.
    Medians count 'not found within the budget' as budget + 1."""
    top = problem.top_set(top_fraction)
    top1 = problem.top_set(0.01)
    sc = problem.true_scores()
    best_possible = float(np.max(sc[np.isfinite(sc)])) if np.isfinite(sc).any() else None
    k = min(HEADLINE_K, len(top))
    by: dict[str, list] = {}
    for t in trajs:
        by.setdefault(t.method, []).append({
            "seed": t.seed,
            "to_k_top5": experiments_to_top(problem, t, top, k),
            "to_first_top5": experiments_to_top(problem, t, top, 1),
            "to_first_top1": experiments_to_top(problem, t, top1, 1),
            "top_found": top_found(problem, t, top),
            "best": float(best_so_far(problem, t)[-1]),
            "failures": sum(s.status == "failed" for s in t.steps),
        })
    out = {"top_size": int(len(top)), "top1_size": int(len(top1)), "headline_k": k, "pool": problem.n, "best_possible": best_possible, "methods": {}}
    budget = max(len(t.steps) for t in trajs)

    def med(rows, key):
        e = np.array([r[key] if r[key] is not None else np.nan for r in rows], float)
        return float(np.median(np.where(np.isfinite(e), e, budget + 1))), int(np.isfinite(e).sum())

    for m, rows in by.items():
        m_k, f_k = med(rows, "to_k_top5")
        m_1, f_1 = med(rows, "to_first_top5")
        m_t1, f_t1 = med(rows, "to_first_top1")
        finite_best = [r["best"] for r in rows if np.isfinite(r["best"])]
        out["methods"][m] = {
            "runs": len(rows),
            "experiments_to_top_median": m_k, "found_top_in_budget": f_k,
            "experiments_to_first_top5_median": m_1, "found_first_top5": f_1,
            "experiments_to_first_top1_median": m_t1, "found_first_top1": f_t1,
            "top_found_mean": float(np.mean([r["top_found"] for r in rows])),
            "best_mean": float(np.mean(finite_best)) if finite_best else None,
            "failures_mean": float(np.mean([r["failures"] for r in rows])),
        }
    return out
