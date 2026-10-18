"""How good are the GP's predictions for a deposition run it has never seen? Leave-one-library-out check (L2).

``calibration.check`` holds out random samples. But samples on one HTEM library (one deposition run,
``Problem.library``) share a substrate, a chamber state and a temperature/pressure setting, so a
random split almost always leaves close neighbours of every test sample in the training set. A new
recipe (the 'suggest recipes' feature, or any real next experiment) is made in a new run, so the
honest test holds out a whole library at a time.

For each demo system and property, and each library L with enough measurements:

* ``all``: fit on every measured sample of the other libraries, predict L.
* ``n=30`` / ``n=60``: fit on a random subset of that size from the other libraries (a campaign's
  worth of data), predict L; a few repetitions.

Each is paired with a **matched random control**: the same training size, and a random test set of
the same size as L drawn from all libraries, so the only difference is whether the test samples
share a deposition run with training samples.

Reported, pooled over every held-out prediction: R², 68%/95% interval coverage with the raw GP and
with the display factor people see (``calibration.display_inflation``), the factor that would have
given 68% coverage, and, per library, how far the sample the model ranks best is from the library's
true best. Results: ``reports/calibration/held_out_libraries.json`` and the evaluation report.
"""

from __future__ import annotations

import functools
import json

import numpy as np

from lodestone import config
from lodestone.loop.calibration import TARGETS, display_inflation
from lodestone.loop.models import Surrogate
from lodestone.loop.problem import transform

MIN_LIBRARY = 10  # smaller held-out libraries are skipped (too few samples for coverage or a 'best')
SIZES = (30, 60)
Z68, Z95 = 1.0, 1.96


def _goal(prob, prop: str):
    """Score (higher is better) for 'the best sample in a library' on this property, in model space."""
    w = prob.spec.window
    if w and w.property == prop:
        return lambda y: -(np.maximum(w.low - y, 0) + np.maximum(y - w.high, 0)), f"closest to {w.low}–{w.high} eV"
    for o in prob.spec.objectives:
        if o.property == prop:
            return (lambda y: y) if o.direction == "max" else (lambda y: -y), o.direction
    for c in prob.spec.constraints:
        if c.property == prop:  # a ≥ constraint: more is better
            return (lambda y: y) if c.op == ">=" else (lambda y: -y), "max" if c.op == ">=" else "min"
    return (lambda y: y), "max"


def _predict(prob, prop: str, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sur = Surrogate(prob.X, "gp", seed=seed).fit({prop: (tr, y[tr], 0)})
    return sur.predict(prob.X[te])[prop]


def _pooled(y: np.ndarray, mu: np.ndarray, sd: np.ndarray, infl: float | None) -> dict:
    z = np.abs(y - mu) / sd
    out = {
        "n_predictions": int(len(y)),
        "r2": float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "rmse": float(np.sqrt(((y - mu) ** 2).mean())),
        "cov68": float((z < Z68).mean()),
        "cov95": float((z < Z95).mean()),
        "needed_scale": float(np.quantile(z, 0.6827)),  # factor on σ that would have given 68% coverage
        "needed_scale95": float(np.quantile(z, 0.95) / Z95),
    }
    if infl:
        out |= {"display_inflation": infl, "cov68_display": float((z / infl < Z68).mean()), "cov95_display": float((z / infl < Z95).mean())}
    return out


def check(sizes=SIZES, reps: int = 5, targets=TARGETS, verbose: bool = True) -> dict:
    from lodestone.loop.benchmark import problem_for

    out: dict = {"config": {"sizes": ["all", *sizes], "reps": reps, "min_library": MIN_LIBRARY}, "results": []}
    z_lib: list[np.ndarray] = []  # pooled held-out-library z per row, to score the recipe factor below (not saved)
    for key, prop in targets:
        prob = problem_for(key)
        y = transform(prop, prob.truth[prop])
        measured = np.flatnonzero(np.isfinite(y))
        libs, counts = np.unique(prob.library[measured], return_counts=True)
        held = [lib for lib, c in zip(libs, counts) if c >= MIN_LIBRARY]
        infl = display_inflation(prob.spec.system).get(prop)
        score, goal = _goal(prob, prop)
        for n in ("all", *sizes):
            arms: dict[str, dict[str, list]] = {a: {"y": [], "mu": [], "sd": []} for a in ("library", "random")}
            per_lib = []
            for li, lib in enumerate(held):
                te = measured[prob.library[measured] == lib]
                rest = measured[prob.library[measured] != lib]
                n_reps = 1 if n == "all" else reps
                for rep in range(n_reps):
                    rng = np.random.default_rng([li, rep, 0 if n == "all" else n])
                    tr = rest if n == "all" else rng.choice(rest, n, replace=False)
                    mu, sd = _predict(prob, prop, y, tr, te, seed=rep)
                    for k, v in (("y", y[te]), ("mu", mu), ("sd", sd)):
                        arms["library"][k].append(v)
                    # matched control: same training size, random test set of the same size, from every library
                    te_r = rng.choice(measured, len(te), replace=False)
                    pool = np.setdiff1d(measured, te_r)
                    tr_r = rng.choice(pool, len(tr), replace=False) if len(tr) < len(pool) else pool
                    mu_r, sd_r = _predict(prob, prop, y, tr_r, te_r, seed=rep)
                    for k, v in (("y", y[te_r]), ("mu", mu_r), ("sd", sd_r)):
                        arms["random"][k].append(v)
                    # does the model's top-ranked sample in the held-out library come close to its real best?
                    s_true, s_pick = score(y[te]), score(y[te][np.argmax(score(mu))])
                    yl = y[te]
                    per_lib.append({
                        "library": str(lib), "rep": rep, "n_test": int(len(te)),
                        "r2": float(1 - ((yl - mu) ** 2).sum() / max(((yl - yl.mean()) ** 2).sum(), 1e-12)),
                        "cov68": float((np.abs(yl - mu) / sd < Z68).mean()),
                        "best_gap": float(s_true.max() - s_pick),  # in property units (decades for conductivity)
                        "random_pick_gap": float(s_true.max() - s_true.mean()),  # expected gap of a random pick
                        "pick_is_best": bool(s_pick >= s_true.max()),
                        "pick_percentile": float((s_true <= s_pick).mean()),  # 1.0 = the library's best
                        "library_range": float(s_true.max() - s_true.min()),
                    })
            row = {"system": prob.spec.system, "property": prop, "campaign": key, "goal": goal, "n_train": n,
                   "libraries": len(held), "libraries_skipped": int(len(libs) - len(held)), "n_measured": int(len(measured))}
            for arm, d in arms.items():
                row[arm] = _pooled(*(np.concatenate(d[k]) for k in ("y", "mu", "sd")), infl)
            L = arms["library"]
            z_lib.append(np.abs(np.concatenate(L["y"]) - np.concatenate(L["mu"])) / np.concatenate(L["sd"]))
            gaps = np.array([p["best_gap"] for p in per_lib])
            row["per_library"] = per_lib
            row["best"] = {
                "median_gap": float(np.median(gaps)), "mean_gap": float(gaps.mean()),
                "median_random_pick_gap": float(np.median([p["random_pick_gap"] for p in per_lib])),
                "pick_is_best": float(np.mean([p["pick_is_best"] for p in per_lib])),
                "median_pick_percentile": float(np.median([p["pick_percentile"] for p in per_lib])),
                "median_library_range": float(np.median([p["library_range"] for p in per_lib])),
                "cov68_min": float(min(p["cov68"] for p in per_lib)), "cov68_max": float(max(p["cov68"] for p in per_lib)),
                "median_library_r2": float(np.median([p["r2"] for p in per_lib])),
            }
            out["results"].append(row)
            if verbose:
                L, R = row["library"], row["random"]
                print(f"{row['system']:11s} {prop:12s} n={n!s:4s} held-out library: R2={L['r2']:.2f} cov68={L['cov68']:.2f} "
                      f"(display {L.get('cov68_display', float('nan')):.2f}) cov95={L['cov95']:.2f} | random: R2={R['r2']:.2f} cov68={R['cov68']:.2f} "
                      f"| best gap median {row['best']['median_gap']:.2f} (random pick {row['best']['median_random_pick_gap']:.2f})", flush=True)
    out["recipe_inflation"] = {s: recipe_inflation_from(out, s) for s in sorted({r["system"] for r in out["results"]})}
    for r, z in zip(out["results"], z_lib):  # score the recipe factor on the system it was *not* learned from
        f = out["recipe_inflation"][r["system"]].get(r["property"])
        if f:
            r["library"] |= {"recipe_inflation": f, "cov68_recipe": float((z / f < Z68).mean()), "cov95_recipe": float((z / f < Z95).mean())}
    return out


def recipe_inflation_from(res: dict, system: str) -> dict[str, float]:
    """Std factor per property for *new recipes* (a new deposition run) on `system`, learned only from the
    *other* systems' held-out-library checks at campaign-sized training sets (median of the factor that
    would have given 68% coverage)."""
    out = {}
    for prop in {r["property"] for r in res["results"]}:
        vals = [r["library"]["needed_scale"] for r in res["results"]
                if r["property"] == prop and r["system"] != system and r["n_train"] != "all"]
        if vals:
            out[prop] = float(np.median(vals))
    return out


def save(**kw) -> dict:
    res = check(**kw)
    path = config.REPORTS_DIR / "calibration"
    path.mkdir(parents=True, exist_ok=True)
    (path / "held_out_libraries.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    _saved.cache_clear()
    return res


@functools.lru_cache(maxsize=1)
def _saved() -> dict | None:
    try:
        return json.loads((config.REPORTS_DIR / "calibration" / "held_out_libraries.json").read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------- what people are shown

# Used when reports/calibration/held_out_libraries.json is missing: recipe_inflation_from() of that
# file for a system not in it (median over every system's campaign-sized checks), as of 2026-10-18.
RECIPE_INFLATION_FALLBACK = {"conductivity": 1.85, "band_gap": 1.85}
RECIPE_INTERVAL_LABEL = "68% interval, calibrated on held-out deposition runs"


def recipe_inflation(system: str) -> dict[str, float]:
    """Std factor per property for the ± shown on *suggested recipes* (never made: a new deposition run),
    learned from the other systems' leave-one-library-out checks. Never narrower than the display factor
    for samples in the data (``calibration.display_inflation``)."""
    res = _saved()
    f = recipe_inflation_from(res, system) if res else dict(RECIPE_INFLATION_FALLBACK)
    base = display_inflation(system)
    return {p: max(f.get(p, 1.0), base.get(p, 1.0)) for p in set(f) | set(base)}
