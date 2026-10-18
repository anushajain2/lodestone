"""Choosing the next experiments (L3, L4).

* ``bayes`` + GP: BoTorch Monte-Carlo acquisition over the discrete pool with sequential greedy
  batches (experiments in flight are "pending", so the batch doesn't pay twice for one idea):
    - maximize:    qLogNoisyExpectedImprovement on the objective
    - constrained: the same, with the constraints (e.g. band gap ≥ 3 eV) as feasibility weights
    - window:      the objective is minus the distance to the target window
    - pareto:      qLogNoisyExpectedHypervolumeImprovement on two objectives
* ``bayes`` + RF: expected improvement from Monte-Carlo draws of the forest's predictive spread.
* ``random``: uniform over untested samples.
* ``grid``: the systematic sweep a person would plan by hand: a regular grid over composition (and
  deposition temperature when it varies), cells visited in raster order from a random start, one
  sample per cell per pass.

Every strategy starts with the same seeded random batch (``initial``), so comparisons are paired.

Library level (``CampaignSpec.experiment_unit = "library"``): one experiment is one deposition run, and
the same three strategies pick whole libraries (``select_*_libraries``; the acquisition is explained
in ``select_bayes_libraries``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling import SobolQMCNormalSampler

from lodestone.loop.calibration import display_inflation
from lodestone.loop.held_out_libraries import recipe_inflation
from lodestone.loop.models import Observations, Surrogate, SuccessModel, training_data
from lodestone.loop.problem import Problem, transform


@dataclass
class Pick:
    index: int
    strategy: str
    acquisition: float | None = None
    prediction: dict[str, tuple[float, float]] = field(default_factory=dict)  # model space mean, std as shown (see std_inflation)
    p_feasible: float | None = None  # from the shown (inflated) std
    std_inflation: dict[str, float] = field(default_factory=dict)  # display-only factor applied to the GP std, per property
    new_run: bool = False  # no training sample from this sample's library (deposition run): std widened by recipe_inflation
    p_success: float | None = None
    notes: list[str] = field(default_factory=list)


def _available(problem: Problem, unavailable: set[int]) -> np.ndarray:
    mask = np.ones(problem.n, dtype=bool)
    if unavailable:
        mask[list(unavailable)] = False
    return np.flatnonzero(mask)


def select_random(problem: Problem, unavailable: set[int], q: int, rng: np.random.Generator) -> list[Pick]:
    avail = _available(problem, unavailable)
    return [Pick(index=int(i), strategy="random") for i in rng.choice(avail, size=min(q, len(avail)), replace=False)]


def _sweep_dims(problem: Problem) -> list[int]:
    dims = [j for j, n in enumerate(problem.feature_names) if n.startswith("x_") or n == "deposition_temp_c"]
    return dims or [0]


def _sweep(Z: np.ndarray, budget: int, rng: np.random.Generator) -> list[int]:
    """Rows of Z in hand-planned sweep order: regular grid, raster order from a random start, one row per cell per pass."""
    lo, hi = Z.min(0), Z.max(0)
    U = (Z - lo) / np.where(hi > lo, hi - lo, 1.0)
    d = U.shape[1]
    bins = max(2, int(np.ceil((1.5 * budget) ** (1.0 / d))))
    cells = np.minimum((U * bins).astype(int), bins - 1)
    members: dict[tuple, list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        members.setdefault(c, []).append(i)
    for v in members.values():
        rng.shuffle(v)
    raster = [c for c in itertools.product(range(bins), repeat=d) if c in members]
    start = int(rng.integers(len(raster)))
    raster = raster[start:] + raster[:start]
    order, depth = [], 0
    while len(order) < len(Z):
        added = False
        for c in raster:
            if depth < len(members[c]):
                order.append(members[c][depth])
                added = True
        if not added:
            break
        depth += 1
    return order


def grid_order(problem: Problem, budget: int, seed: int) -> list[int]:
    """A hand-planned sweep: regular grid over composition (+ temperature), raster order, random start."""
    rng = np.random.default_rng(10_000 + seed)
    return _sweep(problem.X[:, _sweep_dims(problem)], budget, rng)


def select_grid(problem: Problem, unavailable: set[int], q: int, order: list[int]) -> list[Pick]:
    out = []
    for i in order:
        if i not in unavailable:
            out.append(Pick(index=int(i), strategy="grid", notes=["next cell in the planned composition/temperature sweep"]))
            if len(out) == q:
                break
    return out


# ------------------------------------------------------------------------------- Bayesian


def _incumbent(problem: Problem, obs: Observations) -> float:
    """Best observed score (natural-unit values → score), or -inf."""
    if not obs.order:
        return -np.inf
    idx = sorted(obs.tested())
    vals = {p: np.array([obs.values.get(p, {}).get(i, np.nan) for i in idx]) for p in problem.truth}
    sc = problem.score_from_values(vals)
    return float(np.max(sc)) if len(sc) else -np.inf


def _gp_objective_and_constraints(problem: Problem, props: list[str]):
    spec = problem.spec
    if spec.window:
        j = props.index(spec.window.property)
        lo, hi = spec.window.low, spec.window.high
        obj = GenericMCObjective(lambda Y, X=None: -(torch.relu(lo - Y[..., j]) + torch.relu(Y[..., j] - hi)))
    else:
        o = spec.objectives[0]
        j = props.index(o.property)
        sign = 1.0 if o.direction == "max" else -1.0
        obj = GenericMCObjective(lambda Y, X=None: sign * Y[..., j])
    cons = []
    for c in spec.constraints:
        k = props.index(c.property)
        thr = float(transform(c.property, np.array([c.value]))[0])
        cons.append((lambda Y, k=k, thr=thr: thr - Y[..., k]) if c.op == ">=" else (lambda Y, k=k, thr=thr: Y[..., k] - thr))
    return obj, cons


def select_bayes(
    problem: Problem, obs: Observations, unavailable: set[int], pending: list[int], q: int,
    rng: np.random.Generator, surrogate_kind: str = "gp", seed: int = 0,
) -> tuple[list[Pick], dict]:
    spec = problem.spec
    props = spec.properties_needed()
    data = training_data(obs, props)
    if any(len(data[p][0]) < 2 for p in props):
        picks = select_random(problem, unavailable, q, rng)
        for p in picks:
            p.notes.append("not enough measurements yet to fit a model; exploring at random")
        return picks, {"note": "fallback to random"}
    sur = Surrogate(problem.X, surrogate_kind, seed=seed).fit(data)
    succ = SuccessModel(problem.X).fit(obs, props, problem.X)
    avail = _available(problem, unavailable)
    Xa = problem.X[avail]
    pred = sur.predict(Xa)
    p_succ = succ.p_success(Xa)
    log_ps = np.log(p_succ)
    meta = {
        "surrogate": surrogate_kind, "props": props,
        "train_sizes": {p: int(len(data[p][0])) for p in props},
        "censored": {p: int(data[p][2]) for p in props},
        "success_model": succ.clf is not None,
        "features": problem.feature_names,
    }
    chosen: list[int] = []
    picks: list[Pick] = []

    if surrogate_kind == "gp":
        X_base = torch.as_tensor(problem.X[sorted({int(i) for p in props for i in data[p][0]})])
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=int(rng.integers(1 << 30)))
        if spec.kind == "pareto":
            Ytr = np.column_stack([pred[p][0] for p in props])  # only for a reference point
            signs = torch.tensor([1.0 if o.direction == "max" else -1.0 for o in spec.objectives])
            obs_idx = [i for i in obs.tested() if all(i in obs.values.get(p, {}) for p in props)]
            Yobs = np.column_stack([transform(p, np.array([obs.values[p][i] for i in obs_idx])) for p in props]) if obs_idx else Ytr
            Yobs = Yobs * signs.numpy()
            ref = (Yobs.min(0) - 0.1 * (Yobs.max(0) - Yobs.min(0) + 1e-6)).tolist()
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=sur.model, ref_point=ref, X_baseline=X_base, prune_baseline=True, sampler=sampler,
                objective=WeightedMCMultiOutputObjective(weights=signs), cache_root=False,
            )
        else:
            obj, cons = _gp_objective_and_constraints(problem, props)
            acqf = qLogNoisyExpectedImprovement(
                model=sur.model, X_baseline=X_base, objective=obj, constraints=cons or None,
                prune_baseline=True, sampler=sampler, cache_root=False,
            )
        pend = list(pending)
        X_pend = torch.as_tensor(problem.X[pend]) if pend else None
        if X_pend is not None:
            acqf.set_X_pending(X_pend)
        Xc = torch.as_tensor(Xa)
        for _ in range(min(q, len(avail))):
            with torch.no_grad():
                vals = np.concatenate([acqf(Xc[s : s + 512].unsqueeze(1)).numpy() for s in range(0, len(Xc), 512)])
            vals = vals + log_ps
            vals[chosen] = -np.inf
            b = int(np.argmax(vals))
            chosen.append(b)
            cur = torch.as_tensor(Xa[chosen])
            acqf.set_X_pending(cur if X_pend is None else torch.cat([X_pend, cur]))
            picks.append(Pick(index=int(avail[b]), strategy="bayes", acquisition=float(vals[b])))
    else:  # random forest: Monte-Carlo EI from the trees' spread
        S = 256
        draws = {p: pred[p][0][None, :] + pred[p][1][None, :] * rng.standard_normal((S, len(avail))) for p in props}
        nat = {p: (10 ** draws[p] if p == "conductivity" else draws[p]) for p in props}
        nat_full = {p: nat.get(p, np.full((S, len(avail)), np.nan)) for p in problem.truth}
        sc = np.stack([problem.score_from_values({p: nat_full[p][s] for p in nat_full}) for s in range(S)])
        best = _incumbent(problem, obs)
        if np.isfinite(best):
            ei = np.where(np.isfinite(sc), np.maximum(sc - best, 0.0), 0.0).mean(0)
        else:
            ei = np.isfinite(sc).mean(0)  # nothing feasible yet: chase feasibility
        vals = np.log(ei + 1e-12) + log_ps
        for b in np.argsort(-vals)[: min(q, len(avail))]:
            chosen.append(int(b))
            picks.append(Pick(index=int(avail[b]), strategy="bayes", acquisition=float(vals[b])))

    # predictions and feasibility for the explanations. The picks above used the raw GP; what people
    # see uses the std widened to match held-out coverage (loop.calibration.display_inflation), or, for a
    # sample from a library the model has no data from, held-out-library coverage (loop.held_out_libraries).
    infl_seen = {p: f for p, f in display_inflation(spec.system).items() if p in props} if surrogate_kind == "gp" else {}
    infl_new = {p: f for p, f in recipe_inflation(spec.system).items() if p in props} if surrogate_kind == "gp" else {}
    seen_libs = {problem.library[i] for p in props for i in data[p][0]}
    for pk, b in zip(picks, chosen):
        pk.new_run = bool(infl_new) and problem.library[pk.index] not in seen_libs
        pk.std_inflation = infl = infl_new if pk.new_run else infl_seen
        pk.prediction = {p: (float(pred[p][0][b]), float(pred[p][1][b]) * infl.get(p, 1.0)) for p in props}
        pk.p_success = float(p_succ[b])
        if spec.constraints:
            from scipy.stats import norm

            pf = 1.0
            for c in spec.constraints:
                m, s = pk.prediction[c.property]
                thr = float(transform(c.property, np.array([c.value]))[0])
                pf *= norm.sf(thr, m, s) if c.op == ">=" else norm.cdf(thr, m, s)
            pk.p_feasible = float(pf)
    return picks, meta


# ------------------------------------------------------------------------------- library level
#
# A deposition run makes a whole library (~44 spots on a composition gradient at one temperature and
# pressure), so a lab's real decision is which run to do next, not which spot. Here one experiment is
# one library: choosing it reveals every spot on it. Libraries are named by id (``Problem.libraries``).


@dataclass
class LibraryPick:
    library: str
    strategy: str
    acquisition: float | None = None
    notes: list[str] = field(default_factory=list)


def select_random_libraries(problem: Problem, unavailable: set[str], q: int, rng: np.random.Generator) -> list[LibraryPick]:
    avail = [lib for lib in problem.libraries() if lib not in unavailable]
    return [LibraryPick(library=avail[i], strategy="random") for i in rng.choice(len(avail), size=min(q, len(avail)), replace=False)]


def library_grid_order(problem: Problem, budget: int, seed: int) -> list[str]:
    """The run-level sweep a person would plan: the same grid as ``grid_order`` (composition + temperature),
    over each library's mean composition and its deposition temperature, visited in raster order."""
    rng = np.random.default_rng(15_000 + seed)
    libs = problem.libraries()
    names = list(libs)
    Z = np.stack([problem.X[idx][:, _sweep_dims(problem)].mean(0) for idx in libs.values()])
    return [names[i] for i in _sweep(Z, budget, rng)]


def select_grid_libraries(unavailable: set[str], q: int, order: list[str]) -> list[LibraryPick]:
    picks = [LibraryPick(library=lib, strategy="grid", notes=["next run in the planned composition/temperature sweep"]) for lib in order if lib not in unavailable]
    return picks[:q]


def select_bayes_libraries(
    problem: Problem, obs: Observations, unavailable: set[str], pending: list[str], q: int,
    rng: np.random.Generator, seed: int = 0,
) -> tuple[list[LibraryPick], dict]:
    """Rank whole libraries by the batch expected improvement of their spots, taken jointly.

    A library is a q-batch whose members are fixed by the deposition: running it reveals all of its
    spots at once, and the run is worth as much as the best spot it turns up. The exact value of that
    batch is ``E[max(0, max_j f(x_j) - f*)]`` over the library's spots ``x_j``, which is what BoTorch's
    qLogNoisyExpectedImprovement computes when it is given the library's spots as one q-batch (the
    joint GP posterior, Monte-Carlo, constraints as feasibility weights, the window as minus the
    distance). So that is the acquisition, with no new approximation. Two alternatives were rejected:

    * the max of each spot's single-point EI ignores that the run reveals all its spots, so it
      undervalues a library whose spots are each a moderate bet;
    * the sum of single-point EIs (or a greedy q-batch over any spots, then aggregated) treats 44
      neighbouring spots on one gradient as 44 independent tickets, so it overvalues big, flat
      libraries. The joint posterior counts their correlation once.

    It values the best spot only, not how many good spots a run contains; the benchmark reports both
    "runs to the first top-5% sample" (what this optimizes) and "top-5% samples found".

    Libraries still running (``pending``) and earlier picks in the same batch are set as pending
    points, so a batch doesn't pay twice for one idea. The per-spot success model used at spot level
    is not applied here: a library is deposited whole, and conductivity failures already reach the
    GP as censored low values. Pareto (two-objective) campaigns aren't supported at library level:
    hypervolume improvement over a 44-point batch is too expensive to compute exactly.
    """
    spec = problem.spec
    if spec.kind == "pareto":
        raise ValueError("library-level campaigns don't support two objectives")
    props = spec.properties_needed()
    data = training_data(obs, props)
    if any(len(data[p][0]) < 2 for p in props):
        picks = select_random_libraries(problem, unavailable, q, rng)
        for p in picks:
            p.notes.append("not enough measurements yet to fit a model; exploring at random")
        return picks, {"note": "fallback to random"}
    libs = problem.libraries()
    avail = [lib for lib in libs if lib not in unavailable]
    sur = Surrogate(problem.X, "gp", seed=seed).fit(data)
    X_base = torch.as_tensor(problem.X[sorted({int(i) for p in props for i in data[p][0]})])
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=int(rng.integers(1 << 30)))
    obj, cons = _gp_objective_and_constraints(problem, props)
    acqf = qLogNoisyExpectedImprovement(
        model=sur.model, X_baseline=X_base, objective=obj, constraints=cons or None,
        prune_baseline=True, sampler=sampler, cache_root=False,
    )
    pend = [int(i) for lib in pending for i in libs[lib]]
    picks: list[LibraryPick] = []
    for _ in range(min(q, len(avail))):
        if pend:
            acqf.set_X_pending(torch.as_tensor(problem.X[pend]))
        cand = [lib for lib in avail if lib not in {p.library for p in picks}]
        with torch.no_grad():
            vals = np.array([float(acqf(torch.as_tensor(problem.X[libs[lib]]).unsqueeze(0))) for lib in cand])
        b = int(np.argmax(vals))
        picks.append(LibraryPick(library=cand[b], strategy="bayes", acquisition=float(vals[b]),
                                 notes=[f"{len(libs[cand[b]])} spots; highest joint expected improvement of {len(cand)} untried runs"]))
        pend += [int(i) for i in libs[cand[b]]]
    meta = {"surrogate": "gp", "props": props, "train_sizes": {p: int(len(data[p][0])) for p in props}, "features": problem.feature_names}
    return picks, meta
