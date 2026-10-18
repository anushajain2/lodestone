"""Is the GP's uncertainty honest? Held-out calibration check for the surrogate (L2).

For each demo material system and property, fit the GP on n random *measured* samples and predict
every other measured sample of that system. A calibrated model's 68% interval (±1σ) contains the
real measurement 68% of the time, and its 95% interval (±1.96σ) 95% of the time. R² checks the mean.

The same check runs for a few GP variants (``models.GP_OPTIONS``) so the default can be chosen on
evidence. Results: ``reports/calibration/calibration.json`` and the evaluation report.
"""

from __future__ import annotations

import functools
import json

import numpy as np

from lodestone import config
from lodestone.loop.models import Surrogate, gp_options
from lodestone.loop.problem import transform

# (campaign whose problem defines the features, property)
TARGETS = (
    ("zsn-conductivity", "conductivity"),
    ("zsn-gap-window", "band_gap"),
    ("cnzo-transparent-conductor", "conductivity"),
    ("cnzo-transparent-conductor", "band_gap"),
    ("izto-conductivity", "conductivity"),
)

VARIANTS = {
    "default": {},
    "noise floor 0.05": {"noise_floor": 0.05},
    "Matérn-5/2 + lengthscale prior": {"kernel": "matern52"},
    "CV variance scaling": {"calibrate": True},
}


def check(sizes=(30, 60), reps: int = 20, variants: dict[str, dict] | None = None, targets=TARGETS, verbose: bool = True) -> dict:
    from lodestone.loop.benchmark import problem_for

    variants = variants or VARIANTS
    out: dict = {"config": {"sizes": list(sizes), "reps": reps}, "results": []}
    for key, prop in targets:
        prob = problem_for(key)
        y = transform(prop, prob.truth[prop])
        measured = np.flatnonzero(np.isfinite(y))
        for n in sizes:
            if n >= len(measured) - 10:
                continue
            for name, opts in variants.items():
                rng = np.random.default_rng(n)  # same splits for every variant
                acc = {"cov68": [], "cov95": [], "r2": [], "std_scale": [], "needed_scale": []}
                for rep in range(reps):
                    tr = rng.choice(measured, n, replace=False)
                    te = np.setdiff1d(measured, tr)
                    with gp_options(**opts):
                        sur = Surrogate(prob.X, "gp", seed=rep).fit({prop: (tr, y[tr], 0)})
                    mu, sd = sur.predict(prob.X[te])[prop]
                    z = np.abs(y[te] - mu) / sd
                    acc["cov68"].append(float((z < 1.0).mean()))
                    acc["cov95"].append(float((z < 1.96).mean()))
                    acc["r2"].append(float(1 - ((y[te] - mu) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()))
                    acc["std_scale"].append(sur.std_scale.get(prop, 1.0))
                    acc["needed_scale"].append(float(np.quantile(z, 0.6827)))  # extra factor that would give 68% coverage
                row = {
                    "system": prob.spec.system, "property": prop, "n_train": n, "n_measured": int(len(measured)),
                    "variant": name, "options": opts,
                    **{k: float(np.mean(v)) for k, v in acc.items()},
                    "cov68_sem": float(np.std(acc["cov68"], ddof=1) / np.sqrt(reps)),
                }
                out["results"].append(row)
                if verbose:
                    print(
                        f"{row['system']:11s} {prop:12s} n={n:3d} {name:32s} cov68={row['cov68']:.2f} cov95={row['cov95']:.2f} "
                        f"R2={row['r2']:.2f} scale={row['std_scale']:.2f}",
                        flush=True,
                    )
    return out


def loso_inflation(res: dict, system: str) -> dict[str, float]:
    """Fixed std factor per property for `system`, learned only from the *other* systems' held-out
    checks (median over their sizes of the factor that would have given 68% coverage)."""
    out = {}
    for prop in {r["property"] for r in res["results"]}:
        vals = [r["needed_scale"] for r in res["results"] if r["variant"] == "default" and r["property"] == prop and r["system"] != system]
        if vals:
            out[prop] = float(np.median(vals))
    return out


# ------------------------------------------------------------------------- what people are shown

# Used when reports/calibration/calibration.json is missing: loso_inflation() of that file for a
# system not in it (median over all systems' held-out checks), as of 2026-10-18.
DISPLAY_INFLATION_FALLBACK = {"conductivity": 1.54, "band_gap": 2.11}
INTERVAL_LABEL = "68% interval, calibrated on held-out data"


@functools.lru_cache(maxsize=1)
def _saved_results() -> dict | None:
    path = config.REPORTS_DIR / "calibration" / "calibration.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def display_inflation(system: str) -> dict[str, float]:
    """Std factor per property for the GP uncertainty *shown to people* (the ± and feasibility in
    "why this sample", the dashboard and the tool outputs), learned from the other systems' held-out
    checks. Display only: the acquisition keeps the raw GP (``models.GP_OPTIONS``), because inflating
    it there made the loop worse. Properties without a held-out check are absent (shown raw)."""
    res = _saved_results()
    return loso_inflation(res, system) if res else dict(DISPLAY_INFLATION_FALLBACK)


def check_with_loso(**kw) -> dict:
    """The held-out check plus the leave-one-system-out fixed inflation, scored on each system."""
    from lodestone.loop.benchmark import problem_for

    res = check(**kw)
    systems = {problem_for(k).spec.system: k for k, _ in TARGETS}
    for system in systems:
        infl = loso_inflation(res, system)
        targets = tuple(t for t in TARGETS if problem_for(t[0]).spec.system == system)
        sub = check(**{**kw, "targets": targets, "variants": {"fixed inflation (from other systems)": {"std_inflation": infl}}})
        res["results"] += sub["results"]
    return res


def save(**kw) -> dict:
    res = check_with_loso(**kw)
    path = config.REPORTS_DIR / "calibration"
    path.mkdir(parents=True, exist_ok=True)
    (path / "calibration.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


# ------------------------------------------------------------------------------ does it help the loop?

BENCH_CAMPAIGNS = ("zsn-conductivity", "cnzo-transparent-conductor", "zsn-gap-window", "izto-conductivity")


def bench_variants(cal: dict, seeds: int = 20, failure_rate: float = 0.05, verbose: bool = True) -> dict:
    """Bayesian (GP) campaigns with each uncertainty variant, same seeds as the main benchmark, so
    results pair seed by seed with the default."""
    from lodestone.loop.benchmark import problem_for
    from lodestone.loop.engine import experiments_to_top, simulate, summarize_runs, top_found

    out: dict = {"config": {"seeds": seeds, "failure_rate": failure_rate}, "campaigns": {}}
    for key in BENCH_CAMPAIGNS:
        prob = problem_for(key, failure_rate)
        variants = {
            "default": {},
            "CV variance scaling": {"calibrate": True},
            "fixed inflation (from other systems)": {"std_inflation": loso_inflation(cal, prob.spec.system)},
        }
        top = prob.top_set(0.05)
        k = min(5, len(top))
        res: dict = {"variants": {}, "per_seed": {}}
        for name, opts in variants.items():
            trajs = []
            with gp_options(**opts):
                for s in range(seeds):
                    trajs.append(simulate(prob, "bayes", "gp", seed=s))
            res["variants"][name] = {"options": opts, **summarize_runs(prob, trajs)["methods"]["bayes-gp"]}
            res["per_seed"][name] = [
                {"seed": t.seed, "to_k_top5": experiments_to_top(prob, t, top, k), "top_found": top_found(prob, t, top)} for t in trajs
            ]
            if verbose:
                v = res["variants"][name]
                print(f"[{key}] {name:38s} exp-to-{k}-top5 median {v['experiments_to_top_median']:.0f} ({v['found_top_in_budget']}/{seeds}) "
                      f"top5 found {v['top_found_mean']:.1f}", flush=True)
        out["campaigns"][key] = res
    return out


def save_bench(seeds: int = 20) -> dict:
    path = config.REPORTS_DIR / "calibration"
    cal = json.loads((path / "calibration.json").read_text())
    res = bench_variants(cal, seeds=seeds)
    (path / "loop_effect.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res
