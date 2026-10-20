"""A published benchmark: P3HT/CNT blend conductivity from Liang et al. 2021.

Q. Liang et al., "Benchmarking the performance of Bayesian optimization across multiple experimental
materials science domains", npj Computational Materials 7, 188 (2021). Its five datasets are public
(github.com/PV-Lab/Benchmarking, MIT License); P3HT/CNT is the one that shares Lodestone's objective,
electrical conductivity. Data and attribution: ``data/benchmarks/p3ht_cnt/``.

Two protocols:

* **The paper's.** Duplicate recipes averaged (233 rows -> 178 blends), 2 random starting experiments,
  then one experiment per cycle, no simulated failures, 50 seeds. Scored with the paper's own
  metrics: Top%(i), the share of the top-5% blends found after i experiments; the enhancement factor
  EF(i) = Top%_BO(i) / Top%_random(i), with the paper's analytic random baseline (its eqs. 14-15),
  computed per run and summarized by the median; and the acceleration factor at Top% = a, the ratio
  of experiments random needs to experiments BO needs to reach a. (The paper prints AF as i_BO /
  i_random, but describes it as how much faster BO is; we use the ratio that is > 1 when BO is faster.)
* **Lodestone's own.** The same settings as every other campaign in the report (60 experiments, 10
  random seed experiments, batches of 5, 5% simulated failures, 20 seeds), so the result sits beside
  the HTEM campaigns.

What differs from the paper even in its protocol: Lodestone's GP is BoTorch's default (RBF with ARD
and a lengthscale prior) on log10 conductivity with qLogNEI, and its random forest uses Monte-Carlo
expected improvement; the paper's best pairs were GP-ARD (Matérn-5/2) and RF, both with LCB (ratio 2),
on standardized raw conductivity. Its runs continue until the pool is empty; ours stop at
``PAPER_BUDGET`` experiments. The paper gives P3HT/CNT numbers only as a range ('EF_max ... in the 2x
to 5x range' for its four smaller datasets) and in a normalized bar chart (Fig. 4).
"""

from __future__ import annotations

import json
import math

import numpy as np

from lodestone import config
from lodestone.connectors.adapters import LabCSVAdapter
from lodestone.connectors.mapping import load_mapping
from lodestone.loop.engine import simulate, top_found
from lodestone.loop.problem import Problem
from lodestone.schemas import CampaignSpec

DATA = config.DATA_DIR / "benchmarks" / "p3ht_cnt"
PAPER_SEEDS = 50
PAPER_INITIAL = 2
PAPER_BUDGET = 120
PAPER_METHODS = (("random", "gp"), ("bayes", "gp"), ("bayes", "rf"))
AF_LEVELS = (0.5, 0.8)
PAPER_REPORTS = {
    "citation": "Q. Liang et al., npj Computational Materials 7, 188 (2021), doi:10.1038/s41524-021-00656-9",
    "dataset": "P3HT/CNT, 178 points, 5 inputs, electrical conductivity (Table 1)",
    "ef_max_range": "EF_max of the four smaller datasets (P3HT/CNT, AgNP, Perovskite, AutoAM) 'in the 2x to 5x range' (text, Fig. S1)",
    "surrogates": "GP with ARD and RF comparable, both better than GP without ARD (Fig. 4; on P3HT/CNT RF had the highest median EF_max)",
}


def load() -> tuple[list, dict]:
    mapping = load_mapping(DATA / "mapping.yaml")
    recs, reports = LabCSVAdapter(mapping).adapt_csv((DATA / "P3HT_dataset.csv").read_text())
    rep = reports[0]
    return [r.model_dump() for r in recs], {"received": rep.received, "accepted": rep.accepted, "rejected": rep.rejected,
                                            "system": mapping.system}


def problem(recs: list, system: str, budget: int, initial: int, batch: int, failure_rate: float) -> Problem:
    spec = CampaignSpec(name="Most conductive P3HT/CNT blend", system=system, objectives=[{"property": "conductivity"}],
                        budget=budget, initial=initial, batch_size=batch, failure_rate=failure_rate)
    return Problem.from_records(recs, spec)


def random_top_pct(n_pool: int, n_top: int, cycles: int) -> np.ndarray:
    """The paper's analytic random baseline: expected Top% after i draws without replacement (eqs. 14-15)."""
    p, out, cum = [], [], 0.0
    for i in range(1, cycles + 1):
        pi = (n_top - cum) / (n_pool - (i - 1))
        p.append(pi)
        cum += pi
        out.append(cum / n_top)
    return np.array(out)


def top_pct_curve(prob: Problem, traj, top: np.ndarray, cycles: int) -> np.ndarray:
    """Top% after each experiment (in submission order), padded with the last value."""
    top_set = {int(i) for i in top}
    seen, out = set(), []
    for st in traj.steps[:cycles]:
        if st.status == "ok" and st.index in top_set:
            seen.add(st.index)
        out.append(len(seen) / len(top_set))
    out += [out[-1]] * (cycles - len(out))
    return np.array(out)


def first_reach(curve: np.ndarray, a: float) -> float:
    hit = np.flatnonzero(curve >= a - 1e-12)
    return float(hit[0] + 1) if len(hit) else math.inf


def paper_protocol(recs: list, system: str, seeds: int = PAPER_SEEDS, budget: int = PAPER_BUDGET, verbose: bool = True) -> dict:
    prob = problem(recs, system, budget, PAPER_INITIAL, 1, 0.0)
    top = prob.top_set(0.05)
    rand = random_top_pct(prob.n, len(top), budget)
    rand_full = random_top_pct(prob.n, len(top), prob.n)  # random's experiments to a Top% level may exceed our budget
    out = {"pool": prob.n, "top_size": int(len(top)), "budget": budget, "initial": PAPER_INITIAL, "seeds": seeds, "methods": {}}
    for strat, sur in PAPER_METHODS:
        name = strat if strat != "bayes" else f"bayes-{sur}"
        curves = np.stack([top_pct_curve(prob, simulate(prob, strat, sur, seed=s), top, budget) for s in range(seeds)])
        ef = curves / rand  # per run, per cycle
        ef_med = np.median(ef, axis=0)
        i_max = int(np.argmax(ef_med))
        af = {}
        for a in AF_LEVELS:
            i_bo = float(np.median([first_reach(c, a) for c in curves]))
            i_rand = first_reach(rand_full, a)
            af[str(a)] = {"experiments_bo_median": None if math.isinf(i_bo) else i_bo,
                          "experiments_random_expected": None if math.isinf(i_rand) else i_rand,
                          "af": None if math.isinf(i_bo) or math.isinf(i_rand) else i_rand / i_bo}
        out["methods"][name] = {
            "top_pct_median": np.median(curves, axis=0).tolist(),
            "top_pct_q25": np.percentile(curves, 25, axis=0).tolist(),
            "top_pct_q75": np.percentile(curves, 75, axis=0).tolist(),
            "ef_max_median": float(ef_med[i_max]), "ef_max_at_experiment": i_max + 1,
            "ef_max_q25": float(np.percentile(ef[:, i_max], 25)), "ef_max_q75": float(np.percentile(ef[:, i_max], 75)),
            "af": af,
            "top_pct_at": {str(n): float(np.median(curves[:, n - 1])) for n in (20, 40, 60, budget)},
            "ef_of_median_at": {str(n): float(np.median(curves[:, n - 1]) / rand[n - 1]) for n in (20, 40, 60)},
        }
        if verbose:
            m = out["methods"][name]
            print(f"[p3ht paper protocol] {name}: EF_max {m['ef_max_median']:.2f} at {m['ef_max_at_experiment']}; "
                  + "; ".join(f"Top% {a}: {v['experiments_bo_median']} vs random {v['experiments_random_expected']}" for a, v in af.items()), flush=True)
    out["random_analytic_top_pct"] = rand.tolist()
    return out


def lodestone_protocol(recs: list, system: str, seeds: int = 20, verbose: bool = True) -> dict:
    from lodestone.loop.benchmark import METHODS, run_problem

    prob = problem(recs, system, 60, 10, 5, 0.05)
    summary, _, trajs = run_problem(prob, seeds, METHODS, "p3ht", verbose)
    top = prob.top_set(0.05)
    found: dict[str, dict[int, int]] = {}
    for t in trajs:
        found.setdefault(t.method, {})[t.seed] = top_found(prob, t, top)
    for m, by_seed in found.items():
        d = np.array([by_seed[s] - found["random"][s] for s in sorted(by_seed)], float)
        summary["methods"][m]["top_found_minus_random_mean"] = float(d.mean())
        summary["methods"][m]["top_found_minus_random_sem"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    summary["spec"] = prob.spec.model_dump()
    return summary


def run(paper_seeds: int = PAPER_SEEDS, seeds: int = 20, verbose: bool = True) -> dict:
    recs, imp = load()
    out = {"paper": PAPER_REPORTS, "import": imp,
           "paper_protocol": paper_protocol(recs, imp["system"], paper_seeds, verbose=verbose),
           "lodestone_protocol": lodestone_protocol(recs, imp["system"], seeds, verbose)}
    path = config.REPORTS_DIR / "benchmark"
    path.mkdir(parents=True, exist_ok=True)
    (path / "published_p3ht.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out
