"""Simulated campaigns and their scores (E1).

``simulate`` runs a whole campaign in memory (no store) with any strategy. The benchmark and the
"compare with baselines" tool both use it, so the numbers a partner sees for their campaign and the
numbers in the evaluation report come from the same code.

With ``experiment_unit = "library"`` an experiment is one deposition run: choosing a library reveals
every spot on it, and budget, seed batch and batch size count libraries. Every spot still gets its
own ``Step`` (its own real measurement or failure), and ``Step.n`` is the run that revealed it, so
"experiments to a top sample" reads as runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lodestone.loop.models import Observations
from lodestone.loop.problem import Problem
from lodestone.loop.replay import LabReplay, seeded_rng
from lodestone.loop.select import (
    LibraryPick, Pick, grid_order, library_grid_order, select_bayes, select_bayes_libraries, select_grid,
    select_grid_libraries, select_random, select_random_libraries,
)

MAX_RANDOM_RETRIES = 1


@dataclass
class Step:
    n: int  # experiment number (1-based, submission order); the run number at library level
    index: int
    round: int
    status: str = "pending"
    kind: str | None = None
    reason: str | None = None
    library: str | None = None  # library level: the library (run) this spot came from


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
    if spec.experiment_unit == "library":
        return simulate_libraries(problem, strategy, surrogate, seed, budget, batch, initial, failure_rate, delay)
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


def initial_libraries(problem: Problem, q: int, seed: int) -> list[LibraryPick]:
    """The shared seed runs at library level: identical for every strategy with the same seed."""
    return select_random_libraries(problem, set(), q, np.random.default_rng(30_000 + seed))


def next_libraries(problem: Problem, obs: Observations, unavailable: set[str], pending: list[str], q: int,
                   strategy: str, seed: int, round_no: int, grid: list[str] | None = None) -> list[LibraryPick]:
    rng = seeded_rng(seed, "select-library", round_no)
    if strategy == "random":
        return select_random_libraries(problem, unavailable, q, rng)
    if strategy == "grid":
        return select_grid_libraries(unavailable, q, grid or library_grid_order(problem, problem.spec.budget, seed))
    return select_bayes_libraries(problem, obs, unavailable, pending, q, rng, seed=seed + round_no)[0]


def simulate_libraries(problem: Problem, strategy: str, surrogate: str = "gp", seed: int = 0, budget: int = 10,
                       batch: int = 1, initial: int = 2, failure_rate: float = 0.05, delay: int = 0) -> Trajectory:
    """A campaign in which each experiment is a whole library (one deposition run).

    Every spot of a chosen library is replayed: its real measurement, a real missing value, or a
    simulated mishap at ``failure_rate`` (a lost or damaged spot). A spot can't be retried on its own,
    and a library is never run twice, so the budget is at most the number of libraries."""
    if strategy == "bayes" and surrogate != "gp":
        raise ValueError("library-level campaigns use the GP surrogate")
    libs = problem.libraries()
    budget = min(budget, len(libs))
    replay = LabReplay(problem, failure_rate, seed)
    grid = library_grid_order(problem, budget, seed) if strategy == "grid" else None
    traj = Trajectory(method_name(strategy, surrogate), seed)
    obs = Observations()
    in_flight: list[tuple[str, int, int, int]] = []  # (library, run number, submitted round, due round)
    done: set[str] = set()
    used, rnd = 0, 0
    while used < budget or in_flight:
        rnd += 1
        if used < budget:
            q = min(initial if used == 0 else batch, budget - used)
            if used == 0:
                picks = initial_libraries(problem, q, seed)
            else:
                busy = [lib for lib, *_ in in_flight]
                picks = next_libraries(problem, obs, done | set(busy), busy, q, strategy, seed, rnd, grid)
            for pk in picks:
                used += 1
                in_flight.append((pk.library, used, rnd, rnd + delay))
        arrived = [f for f in in_flight if f[3] <= rnd]
        in_flight = [f for f in in_flight if f[3] > rnd]
        for lib, n, sub, _ in arrived:
            done.add(lib)
            for i in libs[lib]:
                out = replay.run(int(i), 1)
                traj.steps.append(Step(n=n, index=int(i), round=sub, status=out.status, kind=out.kind, reason=out.reason, library=lib))
                obs.add(int(i), out.values, out.kind, out.reason)
        if used >= budget and not in_flight:
            break
    traj.steps.sort(key=lambda st: st.n)
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


def per_run(problem: Problem, traj: Trajectory, top: np.ndarray) -> tuple[list[float], list[int]]:
    """Library level: best true score and number of distinct top samples found after each run."""
    sc = problem.true_scores()
    top_set = set(int(i) for i in top)
    best, found, bests, counts = -np.inf, set(), [], []
    for n in sorted({st.n for st in traj.steps}):
        for st in traj.steps:
            if st.n == n and st.status == "ok":
                best = max(best, sc[st.index])
                if st.index in top_set:
                    found.add(st.index)
        bests.append(float(best))
        counts.append(len(found))
    return bests, counts


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
    budget = max(t.steps[-1].n for t in trajs if t.steps)  # experiments (runs at library level) in the longest campaign

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
