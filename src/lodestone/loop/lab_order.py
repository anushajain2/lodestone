"""The lab's own order as a baseline (E1): would the optimizer have beaten what the lab actually did?

Random picks and a grid sweep are weak baselines: no scientist chooses experiments at random. HTEM
dates every library (one deposition run; all its ~44 spots are made at once), so the order the runs
were really made is known, and replaying it is a baseline a lab will recognise.

Design: run level (one experiment = one library), the whole system as the budget, so every method's
full sequence can be compared at any point. Methods: the lab's order; the GP loop started from the
lab's own first two runs (like for like); the GP loop, random and grid from two random seed runs (the
run-level benchmark's design). 5% simulated lost spots, 20 seeds (the lab's order varies only by which
spots are lost).

Caveat, stated in the report: the lab chose its runs for its own goals at the time, which may not be
this campaign's goal, so this is 'the order the runs were made', not 'the scientists' strategy for
this goal'. Only systems with at least ``MIN_DATED`` dated libraries are used.
"""

from __future__ import annotations

import json

import numpy as np

from lodestone import config
from lodestone.loop.benchmark import library_problem, public_records
from lodestone.loop.engine import per_run, simulate_libraries
from lodestone.loop.select import lab_order

CAMPAIGNS = ("zsn-conductivity", "zsn-gap-window", "cnzo-transparent-conductor")
SYSTEMS = ("N-Sn-Zn", "Co-Ni-O-Zn", "In-O-Sn-Zn", "Cu-S-Sn", "Cu-O-Zn", "O-Sn-Ti-Zn", "Cu-S-Sb", "Cu-O-Se-Zn")
MIN_DATED = 20
CHECKPOINTS = (3, 5, 10, 15)
METHODS = (  # (strategy, start, label)
    ("lab-order", "lab", "The lab's actual order"),
    ("bayes", "lab", "Bayesian (GP), from the lab's first 2 runs"),
    ("bayes", "random", "Bayesian (GP), from 2 random runs"),
    ("random", "random", "Random runs"),
    ("grid", "random", "Grid sweep of runs"),
)


def date_coverage() -> dict:
    """Per system: libraries with usable samples, how many are dated, and how many distinct dates."""
    out = {}
    for system in SYSTEMS:
        libs: dict[str, set] = {}
        for r in public_records(system):
            libs.setdefault(str(r["provenance"]["source_library_id"]), set()).add(r["provenance"].get("sample_date"))
        dated = {lib: d - {None} for lib, d in libs.items() if d - {None}}
        dates = sorted({x[:10] for d in dated.values() for x in d})
        out[system] = {"libraries": len(libs), "dated": len(dated), "distinct_dates": len(dates),
                       "first": dates[0] if dates else None, "last": dates[-1] if dates else None}
    return out


def run(seeds: int = 20, verbose: bool = True) -> dict:
    out = {"config": {"seeds": seeds, "initial_runs": 2, "failure_rate": 0.05, "checkpoints": list(CHECKPOINTS), "min_dated": MIN_DATED},
           "coverage": date_coverage(), "campaigns": {}}
    for key in CAMPAIGNS:
        prob = library_problem(key, 0.05, budget=1000, initial=2)
        libs = prob.libraries()
        n = len(libs)
        top = prob.top_set(0.05)
        order = lab_order(prob)
        top_libs = {str(prob.library[i]) for i in top}
        res = {"name": prob.spec.name, "libraries": n, "top_size": int(len(top)), "libraries_with_top": len(top_libs),
               "lab_order": order, "lab_runs_with_top": [k + 1 for k, lib in enumerate(order) if lib in top_libs], "methods": {}}
        for strategy, start, label in METHODS:
            curves = []
            for s in range(seeds):
                traj = simulate_libraries(prob, strategy, "gp", s, budget=n, batch=1, initial=2, failure_rate=0.05, start=start)
                curves.append(per_run(prob, traj, top)[1])
            c = np.array(curves, float)
            first = [int(np.argmax(row > 0)) + 1 if (row > 0).any() else n + 1 for row in c]
            res["methods"][label] = {
                "strategy": strategy, "start": start,
                "runs_to_first_top_median": float(np.median(first)),
                "top_found_mean_at": {str(k): float(c[:, k - 1].mean()) for k in CHECKPOINTS if k <= n},
                "top_found_curves": c.tolist(),
            }
            if verbose:
                m = res["methods"][label]
                print(f"[{key}] {label}: first top run {m['runs_to_first_top_median']:.0f}; found at " +
                      ", ".join(f"{k}: {v:.1f}" for k, v in m["top_found_mean_at"].items()), flush=True)
        lab = np.array(res["methods"][METHODS[0][2]]["top_found_curves"])
        for label, m in res["methods"].items():
            if label == METHODS[0][2]:
                continue
            c = np.array(m["top_found_curves"])
            m["minus_lab_at"] = {}
            for k in CHECKPOINTS:
                if k <= n:
                    d = c[:, k - 1] - lab[:, k - 1]  # paired by seed (same lost spots for the shared runs)
                    m["minus_lab_at"][str(k)] = {"mean": float(d.mean()), "sem": float(d.std(ddof=1) / np.sqrt(len(d)))}
        out["campaigns"][key] = res
    path = config.REPORTS_DIR / "benchmark"
    path.mkdir(parents=True, exist_ok=True)
    (path / "lab_order.json").write_text(json.dumps(out, indent=2))
    return out
