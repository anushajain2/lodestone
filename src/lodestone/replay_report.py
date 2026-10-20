"""Savings report: "run this on your past experiments and see how many you could have skipped".

    uv run lodestone replay-report export.csv --mapping m.yaml \\
        --goal "maximize conductivity with band gap >= 3 eV" --cost-per-experiment 1500

1. The export is imported through its column mapping into a throwaway workspace (same validation and
   reject reasons as a real import).
2. The goal is replayed on the accepted samples with Bayesian optimization, random picks and a grid
   sweep, over several seeds (:func:`lodestone.loop.benchmark.run_problem`, the benchmark's own code).
3. One self-contained HTML page is written: import summary, best-so-far chart, experiments and money
   saved with an interval, and the caveats. Weak or negative results are stated as such.
"""

from __future__ import annotations

import csv
import html
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lodestone.connectors.adapters import LabCSVAdapter
from lodestone.connectors.mapping import ColumnMapping, helix_mapping, load_mapping
from lodestone.goals import find_system, parse_goal
from lodestone.loop.benchmark import _natural, run_problem
from lodestone.loop.engine import experiments_to_top, per_run, simulate, simulate_libraries
from lodestone.loop.problem import Problem, fmt_value
from lodestone.schemas import PROPERTIES, ImportReport

METHODS = (("bayes", "gp"), ("random", "gp"), ("grid", "gp"))
NAMES = {"bayes-gp": "Bayesian optimization", "random": "Random picks", "grid": "Grid sweep"}
MIN_POOL = 30  # fewer accepted samples than this and a replay says nothing useful
MIN_RUNS = 6  # dated deposition runs needed before the run-level comparison with the lab's order can say anything
DEFAULT_BUDGET = 60


class ReplayError(ValueError):
    """The export can't support a replay; the message says why, in plain words."""


# ------------------------------------------------------------------------------------- import


def import_export(text: str, mapping: ColumnMapping, system: str, filename: str) -> tuple[list[dict], list[ImportReport]]:
    """Import into a throwaway workspace (deleted afterwards) exactly as a real onboarding would."""
    from lodestone.onboarding import OnboardingError, import_partner_csv
    from lodestone.store import Database

    with tempfile.TemporaryDirectory(prefix="lodestone-replay-") as d:
        db = Database(Path(d) / "replay.db")
        try:
            db.platform().create_partner("Replay report", "replay")
            scope = db.scope("replay", "replay-report")
            try:
                reports = import_partner_csv(scope, text, system, filename, mapping)
            except OnboardingError as e:
                raise ReplayError(str(e)) from None
            records = scope.list_records(system=system)
        finally:
            db.close()
    return records, reports


def import_summary(reports: list[ImportReport], plan: tuple, mapping: ColumnMapping) -> dict:
    cols, comp, ignored = plan
    reasons: dict[str, dict] = {}
    rows = []
    for r in reports:
        for i in r.issues:
            key = (i.level, i.reason.split(" (")[0])
            reasons.setdefault(key, {"level": i.level, "reason": key[1], "count": 0})["count"] += 1
            rows.append({"library": r.source_library_id, "sample": i.source_sample_id, "level": i.level, "reason": i.reason})
    measured: dict[str, int] = {}
    for r in reports:
        for p, n in r.measurements.items():
            measured[p] = measured.get(p, 0) + n
    return {
        "received": sum(r.received for r in reports),
        "accepted": sum(r.accepted for r in reports),
        "rejected": sum(r.rejected for r in reports),
        "dropped": sum(1 for row in rows if row["level"] == "dropped_measurement"),
        "warnings": sum(1 for row in rows if row["level"] == "warning"),
        "libraries": [{"library": r.source_library_id, "received": r.received, "accepted": r.accepted, "rejected": r.rejected} for r in reports],
        "reasons": sorted(reasons.values(), key=lambda x: -x["count"]),
        "issues": rows,
        "measured": measured,
        "mapping": {"name": mapping.name, "columns": mapping.describe(),
                    "units": {f: c.unit_text for f, c in cols.items() if c.unit_text},
                    "headers": {f: c.header for f, c in cols.items()} | {f"composition:{el}": h for el, h in comp.items()}},
        "ignored_columns": ignored,
    }


# ------------------------------------------------------------------------------------- replay


def _quantiles(curves: list[list[float]], spec) -> dict:
    """Per experiment: 25/50/75th percentile of the best value so far, in natural units.

    Picks actual run values (no averaging), so a point where the median run has not yet found a
    sample meeting the goal stays empty instead of being interpolated."""
    arr = np.array(curves, dtype=float)
    out = {}
    for name, q in (("q25", 0.25), ("q50", 0.5), ("q75", 0.75)):
        v = np.quantile(arr, q, axis=0, method="inverted_cdf")
        nat, _, _ = _natural(spec, np.where(np.isfinite(v), v, np.nan))
        out[name] = [None if not np.isfinite(x) else float(x) for x in nat]
    return out


def _bootstrap(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """90% interval of median(a) − median(b), resampling seeds (paired: same seed batch)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    lo, hi = np.percentile(d, [5, 95])
    return float(lo), float(hi)


def _n(x: float) -> str:
    """Medians of an even number of runs can end in .5: show it rather than round 53.5 and 61 to a gap of 8."""
    x = float(x)
    return f"{x:.0f}".replace("-", "−") if x.is_integer() else f"{x:.1f}".replace("-", "−")


def _verdict(save: dict, bayes: dict, rand: dict, k: int, budget: int, seeds: int) -> tuple[str, str, list[str]]:
    notes = []
    b_frac = bayes["found_top_in_budget"] / bayes["runs"]
    if rand["found_top_in_budget"] < rand["runs"] / 2:
        notes.append(f"Random picks found {k} top-5% samples within {budget} experiments in only {rand['found_top_in_budget']} of "
                     f"{rand['runs']} runs. Runs that never found them count as {budget + 1}, so the true gap could be larger than shown.")
    if seeds < 20:
        notes.append(f"Only {seeds} seeds were run, so the interval is wide and the verdict can change with more runs. Use --seeds 30 for a number to quote.")
    lo, hi = save["interval"]
    if b_frac < 0.5:
        return "weak", (f"Bayesian optimization did not reliably find {k} of the top-5% samples within {budget} experiments "
                        f"(it did in {bayes['found_top_in_budget']} of {bayes['runs']} runs). This replay does not support a savings claim."), notes
    if lo > 0:
        return "positive", (f"Bayesian optimization found {k} of the top-5% samples in a median of {_n(bayes['experiments_to_top_median'])} experiments, "
                            f"against {_n(rand['experiments_to_top_median'])} for random picks: {_n(save['experiments'])} fewer "
                            f"(90% interval {_n(lo)} to {_n(hi)})."), notes
    if hi < 0:
        return "negative", (f"On this data Bayesian optimization was slower than random picks: a median of {_n(bayes['experiments_to_top_median'])} "
                            f"experiments against {_n(rand['experiments_to_top_median'])} (90% interval of the difference {_n(lo)} to {_n(hi)}). "
                            "It would not have saved experiments here."), notes
    return "inconclusive", (f"No clear difference from random picks: Bayesian optimization needed a median of {_n(bayes['experiments_to_top_median'])} "
                            f"experiments, random picks {_n(rand['experiments_to_top_median'])}, and the 90% interval of the saving "
                            f"({_n(lo)} to {_n(hi)}) includes zero. Treat any saving from this data as unproven."), notes


# ------------------------------------------------------------------ the lab's own order

def date_granularity(prob: Problem) -> tuple[str, dict]:
    """'row' when samples carry their own dates, 'run' when every library (batch) has one date,
    'none' when fewer than half the samples are dated or every date is the same."""
    dates = [r["provenance"].get("sample_date") for r in prob.records]
    dated = [d for d in dates if d]
    info = {"dated": len(dated), "samples": prob.n, "distinct_dates": len(set(dated))}
    if len(dated) < 0.5 * prob.n or len(set(dated)) < 2:
        return "none", info
    libs = prob.libraries()
    per_lib = {lib: {dates[i] for i in idx if dates[i]} for lib, idx in libs.items()}
    if len(libs) > 1 and all(len(v) <= 1 for v in per_lib.values()):
        return "run", info | {"runs": sum(1 for v in per_lib.values() if v)}
    rows_sharing = sum(1 for d in dated if dated.count(d) > 1)
    return "row", info | {"rows_sharing_a_date": rows_sharing}


def lab_comparison(prob: Problem, k: int, seeds: int, budget: int) -> dict:
    """The model against the order the lab actually made its samples (or runs), started from the
    lab's own first experiments. Paired by seed; only the simulated failures differ between seeds."""
    level, info = date_granularity(prob)
    out = {"level": level, "info": info}
    if level == "none":
        return out
    top = prob.top_set(0.05)
    if level == "row":
        spec = prob.spec
        runs = {m: [simulate(prob, strat, "gp", seed=s, start="lab") for s in range(seeds)]
                for m, strat in (("lab-order", "lab-order"), ("bayes-gp-from-lab-start", "bayes"))}
        unit, horizon = "experiments", budget
        out["initial"] = spec.initial
    else:
        n_runs = info["runs"]
        initial = 1 if n_runs < 10 else 2
        runs = {m: [simulate_libraries(prob, strat, "gp", s, budget=n_runs, batch=1, initial=initial, failure_rate=prob.spec.failure_rate, start="lab")
                    for s in range(seeds)] for m, strat in (("lab-order", "lab-order"), ("bayes-gp-from-lab-start", "bayes"))}
        unit, horizon = "runs", n_runs
        out["initial"] = initial
        out["found_per_run"] = {m: np.mean([per_run(prob, t, top)[1] for t in ts], axis=0).tolist() for m, ts in runs.items()}
        out["lab_runs_with_top"] = [n + 1 for n, lib in enumerate(dict.fromkeys(st.library for st in runs["lab-order"][0].steps))
                                    if lib in {str(prob.library[i]) for i in top}]
    per_seed = {m: np.array([experiments_to_top(prob, t, top, k) or horizon + 1 for t in ts], float) for m, ts in runs.items()}
    meth = {m: {"to_k_median": float(np.median(v)), "reached": int((v <= horizon).sum()), "runs": len(v),
                "top_found_mean": float(np.mean([len({st.index for st in t.steps if st.status == "ok" and st.index in set(top.tolist())}) for t in ts]))}
            for (m, v), ts in zip(per_seed.items(), runs.values())}
    lo, hi = _bootstrap(per_seed["lab-order"], per_seed["bayes-gp-from-lab-start"])
    saved = meth["lab-order"]["to_k_median"] - meth["bayes-gp-from-lab-start"]["to_k_median"]
    out.update({"unit": unit, "horizon": horizon, "methods": meth, "saved": saved, "interval": [lo, hi],
                "usable": level == "row" or info["runs"] >= MIN_RUNS})
    return out


def _verdict_lab(lab: dict, k: int, seeds: int) -> tuple[str, str]:
    m, u, h = lab["methods"], lab["unit"], lab["horizon"]
    b, l = m["bayes-gp-from-lab-start"], m["lab-order"]
    lo, hi = lab["interval"]
    what = "deposition runs" if u == "runs" else "experiments"
    if not lab["usable"]:
        return "too_few", (f"Your export dates each deposition run, but there are only {lab['info']['runs']} runs: too few to tell whether the model "
                           "would have beaten the order your lab actually made them in. The verdict above is against random picks, which flatters the model.")
    if b["reached"] < b["runs"] / 2:
        return "weak", (f"Started from your lab's own first {what}, the model did not reliably find {k} of the top-5% samples within {h} {what} "
                        f"(it did in {b['reached']} of {b['runs']} runs). This replay does not support a savings claim against what you actually did.")
    if lo > 0:
        return "positive", (f"Against the order your lab actually made them in, the model found {k} of the top-5% samples in a median of {_n(b['to_k_median'])} {what}, "
                            f"against {_n(l['to_k_median'])} for your lab: {_n(lab['saved'])} fewer (90% interval {_n(lo)} to {_n(hi)}).")
    if hi < 0:
        return "negative", (f"Your lab's own order was faster than the model: {_n(l['to_k_median'])} {what} to find {k} of the top-5% samples, "
                            f"against {_n(b['to_k_median'])} for the model started from the same first {what} (90% interval of the difference {_n(lo)} to {_n(hi)}). "
                            "On this data the model would not have saved you experiments.")
    return "inconclusive", (f"No clear difference from what your lab actually did: {_n(b['to_k_median'])} {what} for the model against {_n(l['to_k_median'])} "
                            f"for your lab's order, and the 90% interval of the saving ({_n(lo)} to {_n(hi)}) includes zero.")


def run_replay_report(
    csv_path: str | Path, goal: str, mapping_path: str | Path | None = None, system: str | None = None,
    cost_per_experiment: float = 1500.0, currency: str = "$", seeds: int = 30, budget: int | None = None,
    batch_size: int | None = None, failure_rate: float = 0.05, verbose: bool = False,
) -> dict:
    """Import, replay and summarize. Returns a JSON-serializable dict that :func:`render_html` draws."""
    csv_path = Path(csv_path)
    text = csv_path.read_text(encoding="utf-8-sig")
    mapping = load_mapping(mapping_path) if mapping_path else helix_mapping()
    system = system or mapping.system or find_system(goal)
    if not system:
        raise ReplayError("which material system is this? put `system: In-O-Sn-Zn` in the mapping or pass --system")
    adapter = LabCSVAdapter(mapping)
    try:
        plan = adapter.plan(next(csv.reader(io.StringIO(text)), []))
    except ValueError as e:
        raise ReplayError(f"{csv_path.name}: {e}") from None
    records, reports = import_export(text, mapping, system, csv_path.name)
    imp = import_summary(reports, plan, mapping)
    out: dict = {
        "file": csv_path.name, "goal_text": goal, "system": system, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mapping_file": Path(mapping_path).name if mapping_path else "helix_mapping.yaml (built-in default)",
        "import": imp, "cost_per_experiment": cost_per_experiment, "currency": currency,
    }
    if len(records) < MIN_POOL:
        raise ReplayError(f"only {len(records)} samples were accepted from {csv_path.name} (of {imp['received']} rows); "
                          f"a replay needs at least {MIN_POOL}. Reasons: {[r['reason'] + ' ×' + str(r['count']) for r in imp['reasons'][:5]]}")

    spec, notes = parse_goal(goal, [system])
    if spec.kind == "pareto":
        raise ReplayError("the savings report needs one score: give one objective (optionally with constraints) or a target window")
    pool = len(records)
    if budget is None:
        budget = spec.budget if "budget" in spec.model_fields_set else min(DEFAULT_BUDGET, pool // 2)
    if budget >= pool:
        raise ReplayError(f"a budget of {budget} experiments is not smaller than the {pool} samples in the file; nothing could be skipped")
    upd = {"budget": budget, "failure_rate": failure_rate, "cost_per_experiment": cost_per_experiment, "strategy": "bayes", "seed": 0}
    if batch_size:
        upd["batch_size"] = batch_size
    if spec.initial > budget // 2:
        upd["initial"] = max(1, budget // 2)
    spec = spec.model_validate(spec.model_dump() | upd)
    for p in spec.properties_needed():
        if not imp["measured"].get(p):
            raise ReplayError(f"the goal needs {PROPERTIES[p]['label'].lower()}, but no accepted sample has one. "
                              f"Map a column for it (mapped: {[f for f, _ in imp['mapping']['columns']]})")

    prob = Problem.from_records(records, spec)
    scores = prob.true_scores()
    feasible = int(np.isfinite(scores).sum())
    if feasible < 3:
        raise ReplayError(f"only {feasible} of {pool} samples meet the goal ({spec.summary()}); there is nothing to find")
    summary, curves, trajs = run_problem(prob, seeds, METHODS, tag=csv_path.name, verbose=verbose)
    meth = summary["methods"]
    k = summary["headline_k"]
    top = prob.top_set(0.05)
    per_seed = {m: np.array([experiments_to_top(prob, t, top, k) or budget + 1 for t in trajs if t.method == m], float) for m in meth}
    savings = {}
    for base in ("random", "grid"):
        d = per_seed[base] - per_seed["bayes-gp"]
        exp = meth[base]["experiments_to_top_median"] - meth["bayes-gp"]["experiments_to_top_median"]
        lo, hi = _bootstrap(per_seed[base], per_seed["bayes-gp"])
        savings[base] = {
            "experiments": exp, "interval": [lo, hi], "money": exp * cost_per_experiment, "money_interval": [lo * cost_per_experiment, hi * cost_per_experiment],
            "bayes_faster": int((d > 0).sum()), "tied": int((d == 0).sum()), "bayes_slower": int((d < 0).sum()),
        }
    verdict, sentence, vnotes = _verdict(savings["random"], meth["bayes-gp"], meth["random"], k, budget, seeds)
    lab = lab_comparison(prob, k, seeds, budget)
    headline = "random"
    if lab["level"] != "none":
        lab["verdict"], lab["verdict_text"] = _verdict_lab(lab, k, seeds)
        if lab["usable"]:
            headline = "lab"
            vnotes = [f"Against random picks instead: {sentence}"] + vnotes
            verdict, sentence = lab["verdict"], lab["verdict_text"]
            vnotes.append("Your lab chose its experiments for its own goals at the time, which may not be the goal entered here.")
    _, ylabel, logy = _natural(spec, np.array([0.0]))
    best_nat = _natural(spec, np.array([summary["best_possible"]]))[0][0]
    top_ids = {int(i) for i in top}
    found: dict[str, list[list[int]]] = {}
    for t in trajs:
        hit, row = set(), []
        for st in t.steps:
            if st.status == "ok" and st.index in top_ids:
                hit.add(st.index)
            row.append(len(hit))
        found.setdefault(t.method, []).append(row)
    order = ("bayes-gp", "random", "grid")
    out.update({
        "spec": spec.model_dump(), "goal_summary": spec.summary(), "goal_notes": notes, "pool": pool, "feasible": feasible,
        "top_size": summary["top_size"], "k": k, "budget": budget, "seeds": seeds, "failure_rate": failure_rate,
        "methods": {m: meth[m] | {"name": NAMES[m], "per_seed": [int(x) for x in per_seed[m]]} for m in order},
        "savings": savings, "verdict": verdict, "verdict_text": sentence, "verdict_notes": vnotes, "lab": lab, "headline": headline,
        "best_in_data": {"value": float(best_nat), "text": prob.natural_best(summary["best_possible"])},
        "charts": {
            "best": {"ylabel": ylabel, "unit": ylabel[ylabel.rfind("(") + 1:-1] if ylabel.endswith(")") else "", "log": bool(logy),
                     "series": {m: _quantiles(curves[m], spec) for m in order}, "ref": {"value": float(best_nat), "label": "best in your data"}},
            "found": {"ylabel": f"Top-5% samples found so far (of {summary['top_size']})", "unit": "top-5% samples", "log": False,
                      "series": {m: {f"q{q}": [int(v) for v in np.quantile(np.array(found[m]), q / 100, axis=0, method="inverted_cdf")]
                                     for q in (25, 50, 75)} for m in order},
                      "ref": {"value": float(k), "label": f"target: {k}"}},
        },
        "final_best": {m: _final_best(prob, curves[m]) for m in order},
    })
    return out


def _final_best(prob: Problem, curves: list[list[float]]) -> str:
    v = float(np.quantile(np.array(curves, float)[:, -1], 0.5, method="inverted_cdf"))
    return prob.natural_best(v)


# ------------------------------------------------------------------------------------- render

E = html.escape
COLORS = {"bayes-gp": ("#2a78d6", "#3987e5"), "random": ("#eb6834", "#d95926"), "grid": ("#1baf7a", "#199e70")}


def _money(x: float, cur: str) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}{cur}{abs(x):,.0f}"


def _nice_ticks(lo: float, hi: float, log: bool) -> list[float]:
    """Tick positions in plotted space (log10 values when log)."""
    if log:
        if hi - lo >= 2:
            return [float(t) for t in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
        cands = [np.log10(m * 10.0**e) for e in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for m in (1, 2, 5)]
        return [float(t) for t in cands if lo <= t <= hi]
    span = hi - lo or 1.0
    step = 10 ** np.floor(np.log10(span / 4))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= 6:
            step *= m
            break
    return [float(t) for t in np.arange(np.ceil(lo / step) * step, hi + 1e-12, step)]


def chart_svg(c: dict, budget: int, seeds: int, initial: int = 0, start_label_at: str = "top") -> str:
    """Median line + interquartile band per method against experiments, a dashed reference line, hover target."""
    log = c["log"]
    W, H, L, R, T, B = 860, 300, 70, 150, 14, 44
    ok = (lambda v: v is not None and v > 0) if log else (lambda v: v is not None)  # noqa: E731
    tr = np.log10 if log else (lambda v: v)
    ref = c["ref"]["value"]
    vals = [tr(v) for s in c["series"].values() for q in s.values() for v in q if ok(v)] + ([tr(ref)] if ok(ref) and np.isfinite(ref) else [])
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    pad = (hi - lo) * 0.06 or 0.5
    lo, hi = (lo - pad, hi + pad) if log or lo != 0 else (0.0, hi + pad)
    X = lambda n: L + (n - 1) / max(1, budget - 1) * (W - L - R)  # noqa: E731
    Y = lambda v: T + (hi - tr(v)) / (hi - lo) * (H - T - B)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{E(c["ylabel"])} against experiments run, median over {seeds} seeds per method" class="chart">']
    for t in _nice_ticks(lo, hi, log):
        y = T + (hi - t) / (hi - lo) * (H - T - B)
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{L - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt_value(10**t if log else t)}</text>')
    step = 10 if budget > 30 else 5
    for n in [1] + list(range(step, budget + 1, step)):
        parts.append(f'<text x="{X(n):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{n}</text>')
    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" class="axis"/>')
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" class="axis-label" text-anchor="middle">Experiments run</text>')
    if 0 < initial < budget:  # every method starts from the same random batch, so their curves coincide up to here
        xi = (X(initial) + X(initial + 1)) / 2
        parts.append(f'<line x1="{xi:.1f}" x2="{xi:.1f}" y1="{T}" y2="{H - B}" class="axis"/>'
                     f'<text x="{xi + 6:.1f}" y="{T + 12 if start_label_at == "top" else H - B - 8}" class="tick">← same random start for all</text>')
    if ok(ref) and np.isfinite(ref):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" class="ref"/>'
                     f'<text x="{W - R + 8}" y="{Y(ref) + 4:.1f}" class="ref-label">{E(c["ref"]["label"])}</text>')
    for m in ("grid", "random", "bayes-gp"):
        s = c["series"][m]
        segs, cur = [], []
        for i in range(budget):
            if ok(s["q25"][i]) and ok(s["q75"][i]):
                cur.append(i)
            else:
                segs, cur = (segs + [cur] if cur else segs), []
        segs += [cur] if cur else []
        for seg in segs:
            up = " ".join(f"{X(i + 1):.1f},{Y(s['q75'][i]):.1f}" for i in seg)
            down = " ".join(f"{X(i + 1):.1f},{Y(s['q25'][i]):.1f}" for i in reversed(seg))
            parts.append(f'<polygon points="{up} {down}" class="band s-{m}"/>')
        pts = [(i, s["q50"][i]) for i in range(budget) if ok(s["q50"][i])]
        d, prev = "", None
        for i, v in pts:
            d += ("L" if prev == i - 1 else "M") + f"{X(i + 1):.1f},{Y(v):.1f} "
            prev = i
        parts.append(f'<path d="{d.strip()}" class="line s-{m}"/>')
        if pts:
            i, v = pts[-1]
            parts.append(f'<circle cx="{X(i + 1):.1f}" cy="{Y(v):.1f}" r="4.5" class="dot s-{m}"/>')
    parts.append(f'<line x1="0" x2="0" y1="{T}" y2="{H - B}" class="xhair" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="transparent"/>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6e69;--line:#e4e3df;--grid:#ecebe7;
--s-bayes-gp:#2a78d6;--s-random:#eb6834;--s-grid:#1baf7a;--good:#0b7a3b;--warn:#a35c00;--bad:#b42318;--chip:#f3f2ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#151514;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9a998f;--line:#2e2e2b;--grid:#272725;--s-bayes-gp:#3987e5;--s-random:#d95926;--s-grid:#199e70;--good:#4cc27a;--warn:#e0a040;--bad:#f07068;--chip:#232321}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#151514;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#2e2e2b;--grid:#272725;
--s-bayes-gp:#3987e5;--s-random:#d95926;--s-grid:#199e70;--good:#4cc27a;--warn:#e0a040;--bad:#f07068;--chip:#232321}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:940px;margin:0 auto;padding:32px 16px 64px}h1{font-size:26px;line-height:1.25;margin:0 0 6px}h2{font-size:18px;margin:40px 0 10px}
.sub{color:var(--ink2);margin:0}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-top:16px}
.hero{display:flex;flex-wrap:wrap;gap:28px;align-items:flex-end}.hero .big{font-size:52px;font-weight:650;line-height:1;font-variant-numeric:tabular-nums}
.hero .lab{color:var(--ink2);font-size:14px;margin-top:6px}.verdict{margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid var(--line);font-size:16px}
.hero.unproven .big{font-size:34px;color:var(--ink2)}
.v-too_few{color:var(--warn)}.badge{display:inline-flex;gap:6px;align-items:center;font-weight:600;font-size:13px;padding:2px 10px;border-radius:999px;background:var(--chip);margin-right:8px}
.v-positive{color:var(--good)}.v-inconclusive,.v-weak{color:var(--warn)}.v-negative{color:var(--bad)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.tile .l{color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2);margin-bottom:6px}.key{display:inline-block;width:18px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.key.ref{border-top:1.5px dashed var(--muted)}.chart{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:12px}.axis-label{fill:var(--ink2);font-size:13px}.ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 4}.ref-label{fill:var(--ink2);font-size:12px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.band{opacity:.12;stroke:none}.dot{stroke:var(--card);stroke-width:2}.xhair{stroke:var(--muted);stroke-width:1}
.s-bayes-gp{stroke:var(--s-bayes-gp);fill:var(--s-bayes-gp)}.s-random{stroke:var(--s-random);fill:var(--s-random)}.s-grid{stroke:var(--s-grid);fill:var(--s-grid)}
.line.s-bayes-gp,.line.s-random,.line.s-grid{fill:none}.chartwrap{position:relative}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;min-width:200px}

.tip .row{display:flex;gap:8px;align-items:center;justify-content:space-between}.tip .sw{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}
ul.caveats li{margin:6px 0}.muted{color:var(--muted)}details summary{cursor:pointer;color:var(--ink2);margin-top:10px}code{font-size:13px;background:var(--chip);padding:1px 5px;border-radius:4px}
"""

JS = """
(function(){const d=JSON.parse(document.getElementById('replay-data').textContent);const n=d.budget;
const fmt=v=>{if(v===null)return'none yet';const a=Math.abs(v);return a>=1000?Math.round(v).toLocaleString():a>=1?(+v.toPrecision(3)).toString():(+v.toPrecision(2)).toString()};
document.querySelectorAll('.chartwrap').forEach(wrap=>{const c=d.charts[wrap.dataset.chart];const svg=wrap.querySelector('svg');const hit=wrap.querySelector('.hit');
const xh=wrap.querySelector('.xhair');const tip=wrap.querySelector('.tip');const L=+hit.getAttribute('x'),Wd=+hit.getAttribute('width');
hit.addEventListener('mousemove',ev=>{const pt=svg.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;const p=pt.matrixTransform(svg.getScreenCTM().inverse());
const i=Math.max(0,Math.min(n-1,Math.round((p.x-L)/Wd*(n-1))));const x=L+i/(n-1)*Wd;xh.setAttribute('x1',x);xh.setAttribute('x2',x);xh.setAttribute('visibility','visible');
let h='<div style="font-weight:600;margin-bottom:4px">After '+(i+1)+' experiment'+(i?'s':'')+'</div>';
for(const m of ['bayes-gp','random','grid']){h+='<div class="row"><span><span class="sw" style="background:var(--s-'+m+')"></span>'+d.methods[m].name+
'</span><span>'+fmt(c.series[m].q50[i])+'</span></div>'}tip.innerHTML=h+'<div class="muted" style="margin-top:4px">median run · '+c.unit+'</div>';
const r=wrap.getBoundingClientRect();tip.style.display='block';let left=ev.clientX-r.left+14;if(left+tip.offsetWidth>r.width)left=ev.clientX-r.left-tip.offsetWidth-14;
tip.style.left=Math.max(0,left)+'px';tip.style.top=Math.max(0,ev.clientY-r.top-20)+'px'});
hit.addEventListener('mouseleave',()=>{tip.style.display='none';xh.setAttribute('visibility','hidden')})})})();
"""


def render_html(r: dict) -> str:
    cur = r["currency"]
    imp = r["import"]
    save = r["savings"]["random"]
    saveg = r["savings"]["grid"]
    meth = r["methods"]
    k, budget, seeds = r["k"], r["budget"], r["seeds"]
    lo, hi = save["interval"]
    badge = {"positive": "✓ Clear saving", "inconclusive": "≈ Not proven", "weak": "! Weak result", "negative": "✕ No saving"}[r["verdict"]]
    proven = r["verdict"] == "positive"
    lab = r.get("lab") or {"level": "none", "info": {}}
    vs_lab = r.get("headline") == "lab"
    if vs_lab:
        unit = "deposition runs" if lab["unit"] == "runs" else "experiments"
        h_val, (h_lo, h_hi) = lab["saved"], lab["interval"]
        h_label = f"{unit} saved vs what your lab actually did"
        money = (_money(lab["saved"] * r["cost_per_experiment"], cur), f"at {_money(r['cost_per_experiment'], cur)} per experiment",
                 f"interval {_money(h_lo * r['cost_per_experiment'], cur)} to {_money(h_hi * r['cost_per_experiment'], cur)}") if lab["unit"] == "experiments" else \
                ("—", "no cost per run was entered", "multiply by what one deposition run costs you")
    else:
        h_val, h_lo, h_hi, h_label = save["experiments"], lo, hi, "experiments saved vs random picks"
        money = (_money(save["money"], cur), f"at {_money(r['cost_per_experiment'], cur)} per experiment",
                 f"interval {_money(save['money_interval'][0], cur)} to {_money(save['money_interval'][1], cur)}")

    def tile(v, label):
        return f'<div class="tile"><div class="v">{v}</div><div class="l">{label}</div></div>'

    rows = "".join(
        f"<tr><td><span class='key' style='border-color:var(--s-{m})'></span>{E(meth[m]['name'])}</td>"
        f"<td class='n'>{_n(meth[m]['experiments_to_top_median'])}</td><td class='n'>{meth[m]['found_top_in_budget']}/{meth[m]['runs']}</td>"
        f"<td class='n'>{_n(meth[m]['experiments_to_first_top5_median'])}</td><td class='n'>{meth[m]['top_found_mean']:.1f}</td>"
        f"<td>{E(r['final_best'][m])}</td></tr>"
        for m in ("bayes-gp", "random", "grid")
    )
    libs = "".join(f"<tr><td>{E(x['library'])}</td><td class='n'>{x['received']}</td><td class='n'>{x['accepted']}</td><td class='n'>{x['rejected']}</td></tr>" for x in imp["libraries"])
    level_name = {"rejected": "Row rejected", "dropped_measurement": "Value dropped", "warning": "Warning"}
    reasons = "".join(f"<tr><td>{level_name[x['level']]}</td><td>{E(x['reason'])}</td><td class='n'>{x['count']}</td></tr>" for x in imp["reasons"]) \
        or "<tr><td colspan='3' class='muted'>Every row was accepted with every mapped value.</td></tr>"
    issues = "".join(f"<tr><td>{E(str(x['sample'] or '—'))}</td><td>{E(x['library'])}</td><td>{level_name[x['level']]}</td><td>{E(x['reason'])}</td></tr>" for x in imp["issues"])
    cols = "".join(f"<tr><td><code>{E(f)}</code></td><td>{E(c)}</td><td>{E(imp['mapping']['units'].get(f, ''))}</td></tr>" for f, c in imp["mapping"]["columns"])
    measured = ", ".join(f"{PROPERTIES[p]['label'].lower()} {n}" for p, n in imp["measured"].items()) or "none"
    ignored = ", ".join(f"<code>{E(c)}</code>" for c in imp["ignored_columns"]) or "none"
    notes = "".join(f"<p class='muted'>{E(n)}</p>" for n in r["verdict_notes"])
    goal_notes = "; ".join(n for n in r["goal_notes"] if not n.startswith(("constraint:", "target window:")))
    fr = r["failure_rate"]
    spec = r["spec"]
    caveats = [
        f"<b>This is a replay, not a forecast.</b> The loop could only choose among the {r['pool']} samples your lab already made and measured. "
        "In real use it proposes new recipes, which may turn out better or worse than anything in this file.",
        f"<b>“Top 5%” is relative to your own data:</b> the best {r['top_size']} of the {r['feasible']} samples that meet the goal. "
        "It says nothing about materials outside this file.",
        "<b>One experiment = one row of your export.</b> If your lab makes samples in batches or combinatorial libraries "
        "(for example 44 spots per sputtered library), the real cost is per batch, and the saving does not translate one-to-one.",
        f"<b>Failures are included.</b> A sample whose measurement is missing in your file fails when replayed, as it did in your lab; "
        f"on top of that, {100 * fr:.0f}% of experiments fail at random (simulated mishaps, one retry allowed).",
        f"<b>Numbers are medians over {seeds} seeds</b> (each seed changes the shared starting batch of {spec['initial']} random samples and the "
        f"simulated failures). A run that never found {k} top-5% samples within {budget} experiments counts as {budget + 1}. "
        "The interval is a bootstrap over seeds.",
        ("<b>Your lab's own order is the main comparison.</b> It replays the samples in the order your export dates them (ties in file order), "
         "and the model starts from the same first experiments. Your lab chose its experiments for its own goals at the time, which may not be the goal entered here. "
         "“Random picks” and “Grid sweep” are shown too; they are easier baselines and flatter the model."
         if vs_lab else
         "<b>The baselines are ours, not your lab's history, and they flatter the model.</b> “Random picks” chooses samples uniformly; “Grid sweep” is a planned "
         "raster over composition and temperature. On the HTEM data in this project, the order a lab actually made its runs was a much tougher baseline. "
         + ("Map a date column and the report compares with your lab's own order." if lab["level"] == "none" else "")),
        f"<b>Money saved = experiments saved × {_money(r['cost_per_experiment'], cur)}</b>, the cost per experiment you entered. "
        "Instrument time, staff time and the cost of running the model are not included.",
        f"Rows that were rejected ({imp['rejected']}) are not in the replay at all; values that were dropped ({imp['dropped']}) count as failed measurements.",
    ]
    legend = ("<div class='legend'>" + "".join(f"<span><span class='key' style='border-color:var(--s-{m})'></span>{E(meth[m]['name'])}</span>" for m in ("bayes-gp", "random", "grid"))
              + "<span><span class='key ref'></span>Reference</span></div>")
    data_json = json.dumps(r).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replay savings report</title><style>{CSS}</style></head><body><main>
<p class="sub">Lodestone replay report · {E(r['file'])} · {E(r['generated_at'])}</p>
<h1>How many of these experiments could you have skipped?</h1>
<p class="sub">Goal: <b>{E(r['goal_text'])}</b><br>Read as: {E(r['goal_summary'].split(';')[0])}{(' · ' + E(goal_notes)) if goal_notes else ''}</p>

<div class="card"><div class="verdict"><span class="badge v-{r['verdict']}">{badge}</span>{E(r['verdict_text'])}</div>
<div class="hero{'' if proven else ' unproven'}">
<div><div class="big">{_n(h_val)}</div><div class="lab">{h_label}{'' if proven else ' (not proven)'}<br>median; 90% interval {_n(h_lo)} to {_n(h_hi)}</div></div>
<div><div class="big">{money[0]}</div><div class="lab">{money[1]}{'' if proven or money[0] == '—' else ' (not proven)'}<br>{money[2]}</div></div>
</div>{notes}
<p class="muted">{'Against random picks: ' + _n(save['experiments']) + ' experiments (' + _money(save['money'], cur) + '), 90% interval ' + _n(lo) + ' to ' + _n(hi) + '. ' if vs_lab else ''}Against the grid sweep: {_n(saveg['experiments'])} experiments ({_money(saveg['money'], cur)}), 90% interval {_n(saveg['interval'][0])} to {_n(saveg['interval'][1])}.
Seed by seed against random picks, Bayesian optimization was faster in {save['bayes_faster']}, tied in {save['tied']} and slower in {save['bayes_slower']} of {seeds} runs.</p></div>
{_lab_card(lab, k)}

<h2>How fast each method finds your best samples</h2>
<p class="sub">Median run (line) and middle half of runs (band) over {seeds} seeds, {budget} experiments each. The savings above come from the same runs: for each run, the experiment at which it had found {k} of the top-5% samples. All methods share the first {spec['initial']} random experiments.</p>
<div class="card">{legend}<div class="chartwrap" data-chart="found">{chart_svg(r['charts']['found'], budget, seeds, spec['initial'])}<div class="tip"></div></div>
<p class="muted" style="margin:6px 0 0">{E(r['charts']['found']['ylabel'])}</p></div>

<h2>Best result found so far</h2>
<p class="sub">The dashed line is the best sample in your file that meets the goal ({E(r['best_in_data']['text'])}).</p>
<div class="card">{legend}<div class="chartwrap" data-chart="best">{chart_svg(r['charts']['best'], budget, seeds, spec['initial'], 'top' if spec['window'] else 'bottom')}<div class="tip"></div></div>
<p class="muted" style="margin:6px 0 0">{E(r['charts']['best']['ylabel'])}</p></div>

<div class="card scroll"><table><thead><tr><th>Method</th><th class="n">Experiments to find {k} of the top-5%<br>(median)</th><th class="n">Runs that found them<br>within {budget}</th>
<th class="n">To the first top-5%<br>(median)</th><th class="n">Top-5% samples found<br>(mean of {r['top_size']})</th><th>Best after {budget}<br>(median run)</th></tr></thead><tbody>{rows}</tbody></table></div>

<h2>What came in from your file</h2>
<div class="tiles">{tile(imp['received'], 'rows read')}{tile(imp['accepted'], 'samples accepted')}{tile(imp['rejected'], 'rows rejected')}{tile(imp['dropped'], 'values dropped as impossible')}</div>
<p class="sub" style="margin-top:12px">Accepted values per property: {E(measured)}. Mapping: <code>{E(r['mapping_file'])}</code> ({E(imp['mapping']['name'])}). Columns not used: {ignored}.</p>
<div class="card scroll"><table><thead><tr><th>Outcome</th><th>Reason</th><th class="n">Count</th></tr></thead><tbody>{reasons}</tbody></table>
{f"<details><summary>Every rejected row and dropped value ({len(imp['issues'])})</summary><table><thead><tr><th>Sample</th><th>Library</th><th>Outcome</th><th>Reason</th></tr></thead><tbody>{issues}</tbody></table></details>" if issues else ""}
<details><summary>Column mapping used</summary><table><thead><tr><th>Field</th><th>Column</th><th>Unit read</th></tr></thead><tbody>{cols}</tbody></table></details>
<details><summary>Libraries</summary><table><thead><tr><th>Library</th><th class="n">Rows</th><th class="n">Accepted</th><th class="n">Rejected</th></tr></thead><tbody>{libs}</tbody></table></details></div>

<h2>How to read this, and what it does not show</h2>
<div class="card"><ul class="caveats">{''.join(f'<li>{c}</li>' for c in caveats)}</ul>
<p class="muted">Replay setup: {r['pool']} samples of {E(r['system'])}, {r['feasible']} meet the goal; budget {budget} experiments, a first batch of {spec['initial']} random samples
shared by all methods, then {spec['batch_size']} per round; Gaussian-process surrogate. The numbers behind this page are embedded in it as JSON (<code>#replay-data</code>).</p></div>
</main><script type="application/json" id="replay-data">{data_json}</script><script>{JS}</script></body></html>"""


def _lab_card(lab: dict, k: int) -> str:
    """The comparison with the order the lab actually made its samples, or why there is none."""
    if lab["level"] == "none":
        return ("<h2>Against what your lab actually did</h2><div class='card'><p>Your export has no usable dates "
                f"({lab['info'].get('dated', 0)} of {lab['info'].get('samples', '?')} samples dated), so the model is compared only with random picks and a sweep. "
                "Those are easy baselines: on the HTEM data in this project, the order a lab actually made its runs matched or beat the model on 2 of 3 goals. "
                "Map a <code>date</code> column to get this comparison.</p></div>")
    m, u = lab["methods"], lab["unit"]
    what = "deposition runs" if u == "runs" else "experiments"
    first = f"first {what[:-1]}" if lab["initial"] == 1 else f"first {lab['initial']} {what}"
    head = (f"Your export dates each deposition run ({lab['info']['runs']} runs), so the comparison is per run: the whole run is made at once. "
            if lab["level"] == "run" else
            f"Your export dates {lab['info']['dated']} of {lab['info']['samples']} samples ({lab['info']['distinct_dates']} distinct dates; samples sharing a date are taken in file order). ")
    rows = "".join(f"<tr><td>{E(n)}</td><td class='n'>{_n(m[key]['to_k_median'])}</td><td class='n'>{m[key]['reached']}/{m[key]['runs']}</td>"
                   f"<td class='n'>{m[key]['top_found_mean']:.1f}</td></tr>"
                   for key, n in (("lab-order", "Your lab's actual order"), ("bayes-gp-from-lab-start", "The model, from your lab's " + first)))
    badge = {"positive": "✓ Model ahead", "inconclusive": "≈ No clear difference", "weak": "! Weak result", "negative": "✕ Your lab ahead", "too_few": "? Too few runs"}[lab["verdict"]]
    extra = (f"<p class='muted'>Runs of yours that contained a top-5% sample: {', '.join('#' + str(x) for x in lab['lab_runs_with_top']) or 'none'}.</p>"
             if lab["level"] == "run" else "")
    return (f"<h2>Against what your lab actually did</h2><div class='card'><div class='verdict'><span class='badge v-{E(lab['verdict'])}'>{badge}</span>{E(lab['verdict_text'])}</div>"
            f"<p class='sub'>{E(head)}The model starts from the same {first} your lab made, then chooses; your lab's order continues as it really happened. "
            "Your lab chose for its own goals at the time, which may not be the goal entered here.</p>"
            f"<div class='scroll'><table><thead><tr><th>Order</th><th class='n'>{what.capitalize()} to find {k} of the top-5%<br>(median)</th>"
            f"<th class='n'>Runs that found them<br>within {lab['horizon']}</th><th class='n'>Top-5% samples found<br>(mean)</th></tr></thead><tbody>{rows}</tbody></table></div>{extra}</div>")


def write_replay_report(out: str | Path, **kwargs) -> tuple[Path, dict]:
    r = run_replay_report(**kwargs)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(r), encoding="utf-8")
    return p, r
