"""Benchmark (E1): Bayesian optimization vs random and grid search, on real HTEM data.

For each demo campaign, every method gets the same budget, batch size, seed experiments and lab
failure rate. Metrics:

* experiments needed to find a top-5% sample (the headline),
* best value found against experiments used (the chart),
* how many of the top-5% samples were found,
* robustness: the same campaign at 0%, 10% and 20% failure rates, and with batch sizes 1, 5 and 10.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from lodestone import config
from lodestone.connectors import htem
from lodestone.connectors.adapters import HTEMAdapter
from lodestone.loop.engine import best_so_far, simulate, summarize_runs
from lodestone.loop.problem import Problem, untransform
from lodestone.schemas import PROPERTIES, CampaignSpec

DEMO_CAMPAIGNS = {
    "zsn-conductivity": CampaignSpec(
        name="Most conductive zinc tin nitride", system="N-Sn-Zn",
        objectives=[{"property": "conductivity"}],
    ),
    "cnzo-transparent-conductor": CampaignSpec(
        name="Transparent p-type conductor (gap ≥ 3 eV)", system="Co-Ni-O-Zn",
        objectives=[{"property": "conductivity"}], constraints=[{"property": "band_gap", "op": ">=", "value": 3.0}],
    ),
    "zsn-gap-window": CampaignSpec(
        name="Band gap in the 1.4–1.6 eV solar window", system="N-Sn-Zn",
        window={"property": "band_gap", "low": 1.4, "high": 1.6},
    ),
    "izto-conductivity": CampaignSpec(
        name="Most conductive indium zinc tin oxide", system="In-O-Sn-Zn",
        objectives=[{"property": "conductivity"}], budget=40,
    ),
}

METHODS = (("random", "gp"), ("grid", "gp"), ("bayes", "gp"), ("bayes", "rf"))
LABELS = {"random": "Random", "grid": "Grid sweep", "bayes-gp": "Bayesian (GP)", "bayes-rf": "Bayesian (random forest)"}
COLORS = {"random": "#8a8f98", "grid": "#d08a2c", "bayes-gp": "#2f6fdf", "bayes-rf": "#18a07a"}


@dataclass
class BenchConfig:
    seeds: int = 20
    campaigns: tuple[str, ...] = tuple(DEMO_CAMPAIGNS)
    methods: tuple[tuple[str, str], ...] = METHODS
    failure_rate: float = 0.05
    robustness_campaign: str = "cnzo-transparent-conductor"
    robustness_seeds: int = 10
    failure_rates: tuple[float, ...] = (0.0, 0.1, 0.2)
    batch_sizes: tuple[int, ...] = (1, 5, 10)
    out_dir: str = field(default_factory=lambda: str(config.REPORTS_DIR / "benchmark"))


_RECORDS: dict[str, list] = {}


def public_records(system: str) -> list[dict]:
    """Every accepted HTEM record of a system from the shipped snapshot (no workspace needed)."""
    if system not in _RECORDS:
        seen: set[str] = set()
        recs = []
        for lib in htem.load_snapshot(system):
            recs += [r.model_dump() for r in HTEMAdapter().adapt_library(lib, seen)[0]]
        _RECORDS[system] = recs
    return _RECORDS[system]


def problem_for(key: str, failure_rate: float | None = None, batch: int | None = None) -> Problem:
    spec = DEMO_CAMPAIGNS[key]
    upd = {}
    if failure_rate is not None:
        upd["failure_rate"] = failure_rate
    if batch is not None:
        upd["batch_size"] = batch
    spec = spec.model_copy(update=upd)
    return Problem.from_records(public_records(spec.system), spec)


def run_campaign(key: str, seeds: int, methods, failure_rate: float, batch: int | None = None, verbose=True) -> tuple[dict, dict]:
    prob = problem_for(key, failure_rate, batch)
    trajs, curves = [], {}
    for strat, sur in methods:
        t0 = time.time()
        for s in range(seeds):
            t = simulate(prob, strat, sur, seed=s)
            trajs.append(t)
            curves.setdefault(t.method, []).append(best_so_far(prob, t).tolist())
        if verbose:
            print(f"[{key}] {strat if strat != 'bayes' else 'bayes-' + sur}: {seeds} seeds in {time.time() - t0:.0f}s", flush=True)
    return summarize_runs(prob, trajs), curves


def run_benchmark(cfg: BenchConfig = BenchConfig(), verbose: bool = True) -> dict:
    out = {"config": asdict(cfg), "campaigns": {}, "robustness": {"failure_rate": {}, "batch_size": {}}}
    curves_all = {}
    for key in cfg.campaigns:
        spec = DEMO_CAMPAIGNS[key]
        summary, curves = run_campaign(key, cfg.seeds, cfg.methods, cfg.failure_rate, verbose=verbose)
        summary["spec"] = spec.model_dump()
        summary["summary"] = spec.summary()
        out["campaigns"][key] = summary
        curves_all[key] = curves
    rob_methods = (("random", "gp"), ("grid", "gp"), ("bayes", "gp"))
    for fr in cfg.failure_rates:
        s, _ = run_campaign(cfg.robustness_campaign, cfg.robustness_seeds, rob_methods, fr, verbose=verbose)
        out["robustness"]["failure_rate"][str(fr)] = s["methods"]
    for b in cfg.batch_sizes:
        s, _ = run_campaign(cfg.robustness_campaign, cfg.robustness_seeds, (("random", "gp"), ("bayes", "gp")), cfg.failure_rate, batch=b, verbose=verbose)
        out["robustness"]["batch_size"][str(b)] = s["methods"]
    path = config.REPORTS_DIR / "benchmark"
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(out, indent=2))
    (path / "curves.json").write_text(json.dumps(curves_all))
    plot(out, curves_all, path)
    return out


def _natural(spec: CampaignSpec, score: np.ndarray) -> tuple[np.ndarray, str, bool]:
    """Convert scores back to a plottable natural quantity."""
    if spec.window:
        return -score, f"Distance outside the {spec.window.low:g}–{spec.window.high:g} eV window (eV)", False
    o = spec.objectives[0]
    sign = 1 if o.direction == "max" else -1
    return untransform(o.property, sign * score), f"Best {PROPERTIES[o.property]['label'].lower()} found ({PROPERTIES[o.property]['unit']})", PROPERTIES[o.property]["log"]


def plot(summary: dict, curves_all: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(curves_all)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.6 * len(keys), 7.8), squeeze=False)
    for col, key in enumerate(keys):
        spec = CampaignSpec(**summary["campaigns"][key]["spec"])
        ax = axes[0][col]
        for m, cs in curves_all[key].items():
            arr = np.array(cs, dtype=float)
            arr[~np.isfinite(arr)] = np.nan
            nat, ylabel, logy = _natural(spec, arr)
            mean = np.nanmedian(nat, axis=0)
            q1, q3 = np.nanpercentile(nat, 25, axis=0), np.nanpercentile(nat, 75, axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=COLORS[m], lw=2, label=LABELS[m])
            ax.fill_between(x, q1, q3, color=COLORS[m], alpha=0.12)
            if logy:
                ax.set_yscale("log")
        best = summary["campaigns"][key]["best_possible"]
        if best is not None:
            nat_best, _, _ = _natural(spec, np.array([best]))
            ax.axhline(nat_best[0], color="#111827", ls=":", lw=1)
        ax.set_title(spec.name, fontsize=10.5)
        ax.set_xlabel("Experiments")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax2 = axes[1][col]
        meth = summary["campaigns"][key]["methods"]
        names = list(meth)
        vals = [meth[m]["experiments_to_top_median"] for m in names]
        ax2.bar([LABELS[m].replace(" (", "\n(") for m in names], vals, color=[COLORS[m] for m in names])
        for i, m in enumerate(names):
            ax2.text(i, vals[i] + 0.8, f"{vals[i]:.0f}\n{meth[m]['found_top_in_budget']}/{meth[m]['runs']} found", ha="center", fontsize=8)
        k = summary["campaigns"][key]["headline_k"]
        ax2.set_ylabel(f"Experiments to find {k} of the top-5% samples\n(median; lower is better)", fontsize=9)
        ax2.set_ylim(0, max(vals) * 1.35)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(axis="x", labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=8)
    seeds = summary["config"]["seeds"]
    fig.suptitle(f"Real HTEM measurements, lab replay: median and interquartile range over {seeds} seeds (dotted line: best sample in the data)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path / "benchmark.png", dpi=140)
    plt.close(fig)
