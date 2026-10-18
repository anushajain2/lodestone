"""Loop benchmark (E1): Bayesian loop vs random and greedy on the same experiment budget.

Ground truth is the catalog's computed k and band gap. Every "experiment" returns the truth plus
measurement noise; scores are computed on the true values of everything tested so far.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from lodestone import config
from lodestone.data.catalog import load_catalog
from lodestone.data.features import reduced_features
from lodestone.loop.metrics import DEFAULT_REF, figure_of_merit, hypervolume_2d
from lodestone.loop.strategies import SelectionContext, select
from lodestone.loop.surrogate import to_objectives
from lodestone.schemas import TargetSpec
from lodestone.targets import filter_pool


@dataclass
class BenchConfig:
    seeds: int = 10
    budget: int = 60
    initial: int = 10
    batch: int = 5
    k_noise_rel: float = 0.06
    gap_noise_eV: float = 0.10
    feature_set: str = "curated+priors"
    strategies: tuple[str, ...] = ("random", "greedy", "bayes", "bayes_cost")
    top_fraction: float = 0.01
    ref: str = "default"  # "default" or "target": acquisition reference point at the k>15, gap>4 target (loosened)


def benchmark_pool(spec: TargetSpec | None = None) -> pd.DataFrame:
    spec = spec or TargetSpec(name="benchmark")
    return filter_pool(load_catalog(), spec)


def run_benchmark(cfg: BenchConfig = BenchConfig(), spec: TargetSpec | None = None, verbose: bool = True) -> pd.DataFrame:
    from lodestone.labs.mock_labs import estimate_costs

    pool = benchmark_pool(spec)
    ids = list(pool.index)
    X = reduced_features(ids, cfg.feature_set).values
    Y_true = to_objectives(pool["k_total"].values, pool["band_gap"].values)
    fom = figure_of_merit(pool["k_total"].values, pool["band_gap"].values)
    n_top = max(1, int(round(cfg.top_fraction * len(ids))))
    top_set = set(np.argsort(-fom)[:n_top])
    hv_max = hypervolume_2d(Y_true, DEFAULT_REF)
    costs = estimate_costs(pool)
    hits_set = set(np.flatnonzero((pool["k_total"].values > 15) & (pool["band_gap"].values > 4)))

    if cfg.ref == "target":
        from lodestone.program import ProgramService

        acq_ref = ProgramService._ref_point(TargetSpec(k_min=15, gap_min_eV=4))
    else:
        acq_ref = DEFAULT_REF.copy()
    rows = []
    for seed in range(cfg.seeds):
        init_rng = np.random.default_rng(10_000 + seed)
        init = init_rng.choice(len(ids), size=cfg.initial, replace=False)
        for strat in cfg.strategies:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)  # GP hyperparameter fitting and posterior sampling draw from torch's RNG
            noise_rng = np.random.default_rng(20_000 + seed)
            tested = list(init)
            t0 = time.time()
            spent = float(costs[tested].sum())

            def observe(idx):
                k = 10 ** Y_true[idx, 0] * (1 + cfg.k_noise_rel * noise_rng.standard_normal(len(idx)))
                g = Y_true[idx, 1] + cfg.gap_noise_eV * noise_rng.standard_normal(len(idx))
                return to_objectives(np.clip(k, 1.0, None), np.clip(g, 0.0, None))

            Y_obs = observe(np.array(tested))
            rnd = 0

            def record():
                t = np.array(tested)
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strat,
                        "round": rnd,
                        "n_experiments": len(t),
                        "hv": hypervolume_2d(Y_true[t]),
                        "hv_frac": hypervolume_2d(Y_true[t]) / hv_max,
                        "top1_recall": len(top_set & set(t)) / n_top,
                        "best_fom": float(fom[t].max()),
                        "hits": len(hits_set & set(t)),
                        "cost": spent,
                    }
                )

            record()
            while len(tested) < cfg.budget:
                rnd += 1
                q = min(cfg.batch, cfg.budget - len(tested))
                ctx = SelectionContext(
                    X_pool=X,
                    tested=np.array(tested),
                    Y_tested=Y_obs,
                    unavailable=np.array(tested),
                    q=q,
                    rng=rng,
                    costs=costs,
                    ref_point=acq_ref,
                )
                picks = select(strat, ctx)
                new = np.array([p.index for p in picks])
                tested.extend(new.tolist())
                spent += float(costs[new].sum())
                Y_obs = np.vstack([Y_obs, observe(new)])
                record()
            if verbose:
                last = rows[-1]
                print(
                    f"seed {seed:2d} {strat:10s} hv={last['hv_frac']:.3f} top1%={last['top1_recall']:.2f} "
                    f"hits={last['hits']} ({time.time() - t0:.1f}s)",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    df.attrs.update({"pool_size": len(ids), "n_top": n_top, "hv_max": hv_max, "n_hits_in_pool": len(hits_set)})
    return df


def experiments_to_reach(df: pd.DataFrame, frac: float = 0.9) -> pd.DataFrame:
    """Per seed and strategy: experiments needed to reach `frac` of the best possible hypervolume."""
    out = []
    for (seed, strat), g in df.groupby(["seed", "strategy"]):
        reached = g[g["hv_frac"] >= frac]
        out.append(
            {
                "seed": seed,
                "strategy": strat,
                "experiments": int(reached["n_experiments"].min()) if len(reached) else np.nan,
            }
        )
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame, cfg: BenchConfig) -> dict:
    final = df[df["n_experiments"] == df["n_experiments"].max()]
    reach = experiments_to_reach(df, 0.9)
    summary = {"config": asdict(cfg), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in df.attrs.items()}}
    per = {}
    for strat, g in final.groupby("strategy"):
        r = reach[reach["strategy"] == strat]["experiments"]
        per[strat] = {
            "hv_frac_mean": float(g["hv_frac"].mean()),
            "hv_frac_std": float(g["hv_frac"].std(ddof=1)) if len(g) > 1 else 0.0,
            "top1_recall_mean": float(g["top1_recall"].mean()),
            "top1_recall_std": float(g["top1_recall"].std(ddof=1)) if len(g) > 1 else 0.0,
            "hits_mean": float(g["hits"].mean()),
            "cost_mean": float(g["cost"].mean()),
            "reached_90pct_hv": int(r.notna().sum()),
            "experiments_to_90pct_hv_median": float(r.median()) if r.notna().any() else None,
            "experiments_to_90pct_hv_mean_reached": float(r.mean()) if r.notna().any() else None,
        }
    summary["final"] = per
    return summary


def save_benchmark(df: pd.DataFrame, cfg: BenchConfig, out_dir=None) -> dict:
    out_dir = out_dir or (config.REPORTS_DIR / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rounds.csv", index=False)
    summary = summarize(df, cfg)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_benchmark(df, out_dir)
    return summary


COLORS = {"random": "#8a8f98", "greedy": "#d08a2c", "bayes": "#2f6fdf", "bayes_cost": "#18a07a"}
LABELS = {"random": "Random", "greedy": "Greedy (surrogate mean)", "bayes": "Bayesian (qLogNEHVI)", "bayes_cost": "Bayesian, cost-aware"}


def plot_benchmark(df: pd.DataFrame, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        ("hv_frac", "Hypervolume (fraction of best possible)"),
        ("top1_recall", "Share of true top 1% (k×Eg) found"),
        ("hits", "Hits found (k > 15 and gap > 4 eV)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        for strat, g in df.groupby("strategy"):
            agg = g.groupby("n_experiments")[col].agg(["mean", "sem"])
            ax.plot(agg.index, agg["mean"], color=COLORS.get(strat), label=LABELS.get(strat, strat), lw=2)
            ax.fill_between(agg.index, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=COLORS.get(strat), alpha=0.15)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Experiments")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    n_seeds = df["seed"].nunique()
    fig.suptitle(f"Loop benchmark: mean ± s.e.m. over {n_seeds} seeds, pool of {df.attrs.get('pool_size', '?')} materials", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark.png", dpi=150)
    plt.close(fig)
