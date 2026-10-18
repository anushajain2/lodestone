"""Batch selection: the Bayesian loop (L2), its cost-aware variant (L4) and the baselines (L3).

All strategies pick from a discrete pool of catalog materials. The Bayesian strategy uses BoTorch's
qLogNEHVI (noisy expected hypervolume improvement, log form) with sequential greedy batch
construction: pick the best candidate, condition on it as "pending", pick the next, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler

from lodestone.loop.metrics import DEFAULT_REF, figure_of_merit, hypervolume_2d
from lodestone.loop.surrogate import Surrogate, from_objectives

STRATEGIES = ("random", "greedy", "bayes", "bayes_cost", "ts_hvi", "ucb")


@dataclass
class Pick:
    index: int  # row in the pool arrays
    acquisition: float | None = None
    mean: np.ndarray | None = None  # objective space (log10 k, gap)
    std: np.ndarray | None = None
    cost: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class SelectionContext:
    X_pool: np.ndarray  # features for every pool material
    tested: np.ndarray  # pool indices with accepted measurements
    Y_tested: np.ndarray  # objective values measured for those (log10 k, gap)
    unavailable: np.ndarray  # indices that cannot be picked (tested, pending, excluded)
    q: int
    rng: np.random.Generator
    ref_point: np.ndarray = field(default_factory=lambda: DEFAULT_REF.copy())
    costs: np.ndarray | None = None  # relative cost per pool material
    pending: np.ndarray | None = None  # pool indices already in the lab queue (results not back yet)
    cost_tolerance: float = 0.2  # "similar gain" = within 20%
    mc_samples: int = 128


def _available(ctx: SelectionContext) -> np.ndarray:
    mask = np.ones(len(ctx.X_pool), dtype=bool)
    mask[ctx.unavailable] = False
    return np.flatnonzero(mask)


def select_random(ctx: SelectionContext) -> list[Pick]:
    avail = _available(ctx)
    idx = ctx.rng.choice(avail, size=min(ctx.q, len(avail)), replace=False)
    return [Pick(index=int(i), notes=["random pick (baseline)"]) for i in idx]


def select_greedy(ctx: SelectionContext, surrogate: Surrogate | None = None) -> list[Pick]:
    """Exploit the surrogate: highest predicted k × Eg, ignoring uncertainty."""
    avail = _available(ctx)
    surrogate = surrogate or Surrogate(ctx.X_pool).fit(ctx.X_pool[ctx.tested], ctx.Y_tested)
    mean, std = surrogate.predict(ctx.X_pool[avail])
    k, gap = from_objectives(mean)
    score = figure_of_merit(k, np.clip(gap, 0, None))
    order = np.argsort(-score)[: ctx.q]
    return [
        Pick(index=int(avail[o]), acquisition=float(score[o]), mean=mean[o], std=std[o], notes=["highest predicted k×Eg"])
        for o in order
    ]


def _acq_values(acqf, Xc: torch.Tensor, chunk: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(Xc), chunk):
            out.append(acqf(Xc[s : s + chunk].unsqueeze(1)).numpy())
    return np.concatenate(out)


def select_bayes(ctx: SelectionContext, surrogate: Surrogate | None = None, cost_aware: bool = False) -> list[Pick]:
    avail = _available(ctx)
    surrogate = surrogate or Surrogate(ctx.X_pool).fit(ctx.X_pool[ctx.tested], ctx.Y_tested)
    X_train = torch.as_tensor(ctx.X_pool[ctx.tested])
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=surrogate.model,
        ref_point=torch.as_tensor(ctx.ref_point).tolist(),
        X_baseline=X_train,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([ctx.mc_samples]), seed=int(ctx.rng.integers(1 << 30))),
        cache_root=False,
    )
    Xc = torch.as_tensor(ctx.X_pool[avail])
    X_inflight = torch.as_tensor(ctx.X_pool[ctx.pending]) if ctx.pending is not None and len(ctx.pending) else None
    if X_inflight is not None:
        acqf.set_X_pending(X_inflight)  # don't pay twice for the gain of experiments already running
    mean, std = surrogate.predict(ctx.X_pool[avail])
    chosen: list[int] = []
    picks: list[Pick] = []
    for _ in range(min(ctx.q, len(avail))):
        vals = _acq_values(acqf, Xc)
        vals[chosen] = -np.inf
        best = int(np.argmax(vals))
        notes = []
        if cost_aware and ctx.costs is not None:
            # "Similar expected gain": within cost_tolerance of the best, in linear acquisition units.
            near = np.flatnonzero(vals >= vals[best] + np.log1p(-ctx.cost_tolerance))
            cheapest = int(near[np.argmin(ctx.costs[avail[near]])])
            if cheapest != best:
                notes.append(
                    f"cost-aware: chosen over a {np.exp(vals[best] - vals[cheapest]):.2f}× higher-gain option "
                    f"that costs {ctx.costs[avail[best]] / ctx.costs[avail[cheapest]]:.1f}× more"
                )
            best = cheapest
        chosen.append(best)
        picks.append(
            Pick(
                index=int(avail[best]),
                acquisition=float(vals[best]),
                mean=mean[best],
                std=std[best],
                cost=None if ctx.costs is None else float(ctx.costs[avail[best]]),
                notes=notes,
            )
        )
        acqf.set_X_pending(Xc[chosen] if X_inflight is None else torch.cat([X_inflight, Xc[chosen]]))
    return picks


def select_ucb(ctx: SelectionContext, surrogate: Surrogate | None = None, beta: float = 1.0) -> list[Pick]:
    """Optimistic greedy: rank by k×Eg evaluated at mean + beta·std of both objectives."""
    avail = _available(ctx)
    surrogate = surrogate or Surrogate(ctx.X_pool).fit(ctx.X_pool[ctx.tested], ctx.Y_tested)
    mean, std = surrogate.predict(ctx.X_pool[avail])
    k, gap = from_objectives(mean + beta * std)
    score = figure_of_merit(k, np.clip(gap, 0, None))
    order = np.argsort(-score)[: ctx.q]
    return [Pick(index=int(avail[o]), acquisition=float(score[o]), mean=mean[o], std=std[o], notes=["optimistic k×Eg (UCB)"]) for o in order]


def select_ts_hvi(ctx: SelectionContext, surrogate: Surrogate | None = None) -> list[Pick]:
    """Multi-objective Thompson sampling: for each batch slot, draw one joint posterior sample over
    the pool and pick the material whose sampled values add the most hypervolume to the current
    front (measured points plus the samples already chosen for this batch)."""
    avail = _available(ctx)
    surrogate = surrogate or Surrogate(ctx.X_pool).fit(ctx.X_pool[ctx.tested], ctx.Y_tested)
    Xa = torch.as_tensor(ctx.X_pool[avail])
    with torch.no_grad():
        post = surrogate.model.posterior(Xa)
        torch.manual_seed(int(ctx.rng.integers(1 << 30)))
        samples = post.rsample(torch.Size([ctx.q])).numpy()  # (q, n, 2)
    mean, std = surrogate.predict(ctx.X_pool[avail])
    front = ctx.Y_tested.copy()
    if ctx.pending is not None and len(ctx.pending):
        front = np.vstack([front, mean[np.isin(avail, ctx.pending)]]) if np.isin(avail, ctx.pending).any() else front
    chosen: list[int] = []
    picks: list[Pick] = []
    for j in range(min(ctx.q, len(avail))):
        f = samples[j]
        base = hypervolume_2d(front, ctx.ref_point)
        dominated = np.zeros(len(f), dtype=bool)
        for y in front:
            dominated |= np.all(f <= y, axis=1)
        gains = np.zeros(len(f))
        for i in np.flatnonzero(~dominated):
            gains[i] = hypervolume_2d(np.vstack([front, f[i]]), ctx.ref_point) - base
        gains[chosen] = -np.inf
        if not np.any(gains > 0):
            w = ctx.rng.dirichlet([1, 1])  # nothing improves the sampled front: random scalarization
            z = (f - f.mean(0)) / (f.std(0) + 1e-9)
            gains = z @ w
            gains[chosen] = -np.inf
        best = int(np.argmax(gains))
        chosen.append(best)
        front = np.vstack([front, f[best]])
        picks.append(Pick(index=int(avail[best]), acquisition=float(gains[best]), mean=mean[best], std=std[best], notes=[]))
    return picks


def select(strategy: str, ctx: SelectionContext, surrogate: Surrogate | None = None) -> list[Pick]:
    if strategy == "random" or len(ctx.tested) < 2:
        return select_random(ctx)
    if strategy == "greedy":
        return select_greedy(ctx, surrogate)
    if strategy == "bayes":
        return select_bayes(ctx, surrogate)
    if strategy == "bayes_cost":
        return select_bayes(ctx, surrogate, cost_aware=True)
    if strategy == "ts_hvi":
        return select_ts_hvi(ctx, surrogate)
    if strategy == "ucb":
        return select_ucb(ctx, surrogate)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
