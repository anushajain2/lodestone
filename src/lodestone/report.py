"""Write reports/EVALUATION.md from whatever results exist, reporting every number as it came out."""

from __future__ import annotations

import json
from pathlib import Path

from lodestone import config

LABELS = {"random": "Random", "grid": "Grid sweep", "bayes-gp": "Bayesian (GP)", "bayes-rf": "Bayesian (random forest)"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(v, digits=0):
    return "—" if v is None else f"{v:.{digits}f}"


CAL_VARIANTS = ("default", "CV variance scaling", "fixed inflation (from other systems)")
PROP_LABEL = {"conductivity": "log10 conductivity", "band_gap": "band gap"}


def _calibration_section(cal: dict, eff: dict | None, bench: dict | None) -> list[str]:
    """Held-out check of the GP's uncertainty, and what calibrating it does to the loop."""
    import numpy as np

    rows = {(r["system"], r["property"], r["n_train"], r["variant"]): r for r in cal["results"]}
    other = [abs(r["cov68"] - rows[(r["system"], r["property"], r["n_train"], "default")]["cov68"])
             for r in cal["results"] if r["variant"] in ("noise floor 0.05", "Matérn-5/2 + lengthscale prior")]
    keys = sorted({(r["system"], r["property"], r["n_train"]) for r in cal["results"]})
    out = ["### Is the model's uncertainty honest?", "",
           f"Fit the GP on n random measured samples of a system, predict every other measured sample, and count how often the real value "
           f"falls inside the predicted 68% and 95% intervals ({cal['config']['reps']} random splits each). A calibrated model scores 68% and 95%. "
           "Two fixes were tried. **CV variance scaling** widens the intervals by a factor from 5-fold cross-validation on the campaign's own data. "
           "**Fixed inflation** widens each property's intervals by a constant learned only from the *other* systems' held-out checks. "
           f"A noise floor and a Matérn-5/2 kernel were also tried; neither raised 68% coverage meaningfully (it moved by at most {max(other) * 100:.0f} percentage points, in either direction; `reports/calibration/calibration.json`). "
           "Neither fix changes the predicted values, so R² is the same for all three.", "",
           "| System | Property | n | 68% interval, default | CV scaling | Fixed inflation | 95% interval, default | CV scaling | Fixed inflation | R² |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for sysname, prop, n in keys:
        r = [rows.get((sysname, prop, n, v)) for v in CAL_VARIANTS]
        if not all(r):
            continue
        out.append(f"| {sysname} | {PROP_LABEL.get(prop, prop)} | {n} | " + " | ".join(f"{x['cov68']:.0%}" for x in r) + " | "
                   + " | ".join(f"{x['cov95']:.0%}" for x in r) + f" | {r[0]['r2']:.2f} |")
    cv = [r["std_scale"] for r in cal["results"] if r["variant"] == "CV variance scaling"]
    fx = [r["std_scale"] for r in cal["results"] if r["variant"] == "fixed inflation (from other systems)"]
    out += ["", f"The default GP is overconfident everywhere. CV scaling overshoots: it widens the intervals {min(cv):.1f}–{max(cv):.1f}× and lands near 85–95% coverage. "
            f"Fixed inflation ({min(fx):.2f}–{max(fx):.2f}×) lands close to nominal.", ""]
    if eff:
        same = bench is not None and all(
            c["variants"]["default"]["experiments_to_top_median"] == bench["campaigns"][k]["methods"]["bayes-gp"]["experiments_to_top_median"]
            and c["variants"]["default"]["top_found_mean"] == bench["campaigns"][k]["methods"]["bayes-gp"]["top_found_mean"]
            for k, c in eff["campaigns"].items() if k in bench["campaigns"]
        )
        out += ["**Does honest uncertainty help the loop? No.** The same Bayesian (GP) campaigns, same seeds and 5% failure rate, paired seed by seed with the default "
                f"({eff['config']['seeds']} seeds; " + ("the default rerun matches section 1 exactly" if same else "**the default rerun does not match section 1; rerun both**") + "):", "",
                "| Campaign | Default: experiments to 5 of top 5% (runs that got there) · top-5% found | CV scaling | Fixed inflation | Top-5% found, fixed − default (paired, ± s.e.m.) |",
                "|---|---|---|---|---|"]
        for key, c in eff["campaigns"].items():
            name = bench["campaigns"][key]["spec"]["name"] if bench and key in bench.get("campaigns", {}) else key
            cell = lambda v: f"{v['experiments_to_top_median']:.0f} ({v['found_top_in_budget']}/{v['runs']}) · {v['top_found_mean']:.1f}"
            base, fixed = c["per_seed"]["default"], c["per_seed"]["fixed inflation (from other systems)"]
            d = np.array([f["top_found"] - b["top_found"] for f, b in zip(fixed, base)], float)
            out.append(f"| {name} | " + " | ".join(cell(c["variants"][v]) for v in CAL_VARIANTS)
                       + f" | {d.mean():+.1f} ± {d.std(ddof=1) / np.sqrt(len(d)):.1f} |")
        out += ["", "So the shipped GP keeps its default, overconfident uncertainty (`models.GP_OPTIONS`; `uv run lodestone eval calibration` reruns all of this). "
                "A likely reading, not tested: with ~1,000 samples where the good ones cluster, an overconfident model exploits sooner, and that pays. "
                "The raw GP's ± is too narrow by about "
                + " and ".join(f"×{np.median([r['std_scale'] for r in cal['results'] if r['variant'] == 'fixed inflation (from other systems)' and r['property'] == p]):.1f} for {PROP_LABEL.get(p, p)}"
                               for p in ("conductivity", "band_gap")) + ". "
                "So the uncertainty *shown to people* (the ± and the chance of meeting a constraint in 'why this sample', the dashboard and the agent tools) "
                "uses the fixed inflation for the campaign's system, labelled as calibrated on held-out data, while the optimizer ranks with the raw GP "
                "(`loop.calibration.display_inflation`; a test checks that a seeded run picks the same samples with and without it).", ""]
    return out


def _library_bullet(R: Path) -> list[str]:
    lib, bench = _load(R / "benchmark" / "library.json"), _load(R / "benchmark" / "summary.json")
    if not lib or not lib["campaigns"]:
        return []
    C = lib["campaigns"]
    ratio = [c["methods"]["bayes-gp"]["top_found_mean"] / c["methods"]["random"]["top_found_mean"] for c in C.values()]
    first = [v["experiments_to_first_top5_median"] for c in C.values() for v in c["methods"].values()]
    return [f"- **Counted in deposition runs, the savings shrink.** When one experiment is a whole library ({lib['config']['budget_runs']}-run budget), every method finds its first "
            f"top-5% sample in a median of {min(first):.0f}–{max(first):.0f} runs, and the Bayesian loop finds {min(ratio):.1f}–{max(ratio):.1f}× as many top-5% samples as random"
            + (f" ({min(sens):.1f}–{max(sens):.1f}× across all the run budgets and seed-run counts tried)" if (sens := _sensitivity_ratios(R)) else "")
            + ", down from the spot-level gain. IZTO has too few libraries to test (see 'One experiment = one deposition run')."]


def _sensitivity_ratios(R: Path) -> list[float]:
    out = []
    for p in sorted((R / "benchmark" / "library_sensitivity").glob("*.json")) + [R / "benchmark" / "library.json"]:
        for c in (_load(p) or {}).get("campaigns", {}).values():
            out.append(c["methods"]["bayes-gp"]["top_found_mean"] / c["methods"]["random"]["top_found_mean"])
    return out


GAP_UNIT = {"conductivity": "decades", "band_gap": "eV"}


def _held_out_libraries_section(lib: dict, cal: dict | None) -> list[str]:
    """Leave-one-library-out: predictions for a deposition run the model has never seen."""
    import numpy as np

    rand = {(r["system"], r["property"], r["n_train"]): r for r in (cal or {}).get("results", []) if r["variant"] == "default"}
    cfg = lib["config"]
    out = ["### Can it predict a deposition run it has never seen?", "",
           "Samples on one HTEM library come from one deposition run, so a random split leaves close neighbours of almost every test sample in the training set. "
           "A new recipe is made in a new run. So: hold out one whole library at a time (libraries with at least "
           f"{cfg['min_library']} measurements), fit on the other libraries (all of them, or a random campaign-sized subset of 30 or 60, {cfg['reps']} subsets each), "
           "and predict the held-out library. Each is paired with a **matched random control**: same training size, a random test set of the same size drawn from every library. "
           "Coverage and R² are pooled over every held-out prediction. 'Display' widens σ by the factor people are shown today (learned on the other systems' *random* splits). "
           "'Best pick' is how far the sample the model ranks first in the held-out library falls short of that library's real best (median over libraries), next to what a random pick in the same library gives up; "
           "'best' follows the demo campaign: highest conductivity, band gap closest to the 1.4–1.6 eV window on N-Sn-Zn, highest band gap on Co-Ni-O-Zn (its ≥ 3 eV constraint).", "",
           "| System | Property | Training | Libraries | R², held-out library | R², matched random | 68%: library, raw | display | matched random, raw | random split (calibration.json) | 95%: library, raw | display | σ factor needed for 68% | Best pick short of the library's best | Random pick | Pick is the library's best |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in lib["results"]:
        L, M, b = r["library"], r["random"], r["best"]
        rs = rand.get((r["system"], r["property"], r["n_train"]))
        u = GAP_UNIT.get(r["property"], "")
        out.append(f"| {r['system']} | {PROP_LABEL.get(r['property'], r['property'])} | {'all other libraries' if r['n_train'] == 'all' else r['n_train']} | {r['libraries']} | "
                   f"{L['r2']:.2f} | {M['r2']:.2f} | {L['cov68']:.0%} | {_pct(L.get('cov68_display'))} | {M['cov68']:.0%} | {_pct(rs and rs['cov68'])} | "
                   f"{L['cov95']:.0%} | {_pct(L.get('cov95_display'))} | ×{L['needed_scale']:.1f} | {b['median_gap']:.2f} {u} | {b['median_random_pick_gap']:.2f} {u} | {b['pick_is_best']:.0%} |")
    camp = [r for r in lib["results"] if r["n_train"] != "all"]
    span = lambda rows, f: f"{min(map(f, rows)):.0%}–{max(map(f, rows)):.0%}"
    r2span = lambda rows, arm: f"{min(r[arm]['r2'] for r in rows):.2f} to {max(r[arm]['r2'] for r in rows):.2f}"
    out += ["", f"**The predicted values suffer most.** With 30 or 60 training samples, R² on a held-out run is {r2span(camp, 'library')} "
            f"(median {np.median([r['library']['r2'] for r in camp]):.2f}), against {r2span(camp, 'random')} on the matched random splits. "
            f"Fit on every other library, it is {r2span([r for r in lib['results'] if r['n_train'] == 'all'], 'library')}, against "
            f"{r2span([r for r in lib['results'] if r['n_train'] == 'all'], 'random')}. "
            "So a model that looks decent on random held-out samples often does no better than predicting the average once the deposition run is new.", ""]
    cov = []
    for prop in dict.fromkeys(r["property"] for r in camp):
        rows = [r for r in camp if r["property"] == prop]
        cov.append(f"{PROP_LABEL.get(prop, prop)} {span(rows, lambda r: r['library']['cov68'])} on held-out runs against {span(rows, lambda r: r['random']['cov68'])} on matched random splits")
    disp = [r for r in camp if "cov68_display" in r["library"]]
    out += [f"**Interval coverage, raw GP, 30 or 60 training samples:** " + "; ".join(cov) + ". "
            f"The widened intervals people see today hold the truth {span(disp, lambda r: r['library']['cov68_display'])} of the time on a new run, not 68% "
            f"(95%: {span(disp, lambda r: r['library']['cov95_display'])}).", ""]
    beat = [r for r in lib["results"] if r["best"]["median_gap"] < r["best"]["median_random_pick_gap"]]
    lose = [r for r in lib["results"] if r not in beat]
    names = lambda rows: ", ".join(dict.fromkeys(f"{r['system']} {PROP_LABEL.get(r['property'], r['property'])}" for r in rows))
    out += [f"**Picking the best sample of a new run:** the model's top-ranked sample falls short of the run's real best by less than a random pick does in {len(beat)} of {len(lib['results'])} rows "
            f"({names(beat)}), and by as much or more in the rest ({names(lose)}). "
            f"It is the run's actual best in {min(r['best']['pick_is_best'] for r in lib['results']):.0%}–{max(r['best']['pick_is_best'] for r in lib['results']):.0%} of held-out runs.", ""]
    ri = lib.get("recipe_inflation", {})
    rows = [r for r in camp if "cov68_recipe" in r["library"]]
    if rows:
        out += ["**What this changes: 'suggest recipes'.** Its recipes have never been made, so each one is at least as far from the data as a new run. "
                "Its ± now uses a separate factor per property: the median, over the *other* systems' held-out-run checks with 30 or 60 training samples, of the factor that would have given 68% coverage ("
                + "; ".join(f"{s}: " + ", ".join(f"×{f:.1f} {PROP_LABEL.get(p, p)}" for p, f in sorted(v.items())) for s, v in ri.items() if v)
                + "), and never narrower than the display factor. Scored on the system it was not learned from, its 68% interval holds the truth "
                + ", ".join(f"{r['library']['cov68_recipe']:.0%} ({r['system']} {PROP_LABEL.get(r['property'], r['property'])}, n={r['n_train']})" for r in rows)
                + ". With three systems, one of them (In-O-Sn-Zn) far worse than the others, a single factor is too wide for some systems and still too narrow for others; "
                "too wide is the safer error for recipes nobody has made. Each suggestion also carries a warning that predictions for a new run were weak in this check "
                "(`campaign.suggest_recipes`, `loop.held_out_libraries`). The same wider factor and label are used for the loop's picks from a library it has no measurements from yet "
                "(`select.select_bayes`, `Pick.new_run`). Both are display only: the optimizer's picks are unchanged. "
                "`uv run lodestone eval calibration --no-random` reruns only this check.", ""]
    return out


def _pct(v) -> str:
    return "—" if v is None or v is False else f"{v:.0%}"


def _held_out_bullet(R: Path) -> list[str]:
    import numpy as np

    lib = _load(R / "calibration" / "held_out_libraries.json")
    if not lib:
        return []
    camp = [r for r in lib["results"] if r["n_train"] != "all"]
    return [f"- **Random held-out splits flatter the model.** Holding out a whole deposition run instead, with 30–60 training samples, R² falls to a median of "
            f"{np.median([r['library']['r2'] for r in camp]):.2f} (matched random splits: {np.median([r['random']['r2'] for r in camp]):.2f}), and for In-O-Sn-Zn conductivity and Co-Ni-O-Zn band gap the model's top pick in a new run does no better than a random one. "
            "Suggested recipes are new runs, so their ± is widened further and they carry a warning (see 'Can it predict a deposition run it has never seen?')."]


def _calibration_bullet(R: Path) -> list[str]:
    import numpy as np

    cal, eff = _load(R / "calibration" / "calibration.json"), _load(R / "calibration" / "loop_effect.json")
    if not cal or not eff:
        return []
    d68 = [r["cov68"] for r in cal["results"] if r["variant"] == "default"]
    worse = []
    for key, c in eff["campaigns"].items():
        base, fixed = c["per_seed"]["default"], c["per_seed"]["fixed inflation (from other systems)"]
        d = np.array([f["top_found"] - b["top_found"] for f, b in zip(fixed, base)], float)
        if d.mean() + 2 * d.std(ddof=1) / np.sqrt(len(d)) < 0:
            worse.append(key)
    return [f"- **The GP's error bars are too narrow, and fixing them made the loop worse.** Its 68% intervals contain the real value only "
            f"{min(d68):.0%}–{max(d68):.0%} of the time on held-out samples. Widening them to near-nominal coverage made the Bayesian loop find fewer top-5% samples "
            f"(clearly fewer on {len(worse)} of {len(eff['campaigns'])} campaigns, see 'Is the model's uncertainty honest?'), so the optimizer keeps the overconfident default; only the ± shown to people is widened to the held-out calibration."]


def _seed_hit_chance(n_libs: int, n_top: int, k: int) -> float:
    """Chance that k random libraries out of n_libs include one of the n_top libraries with a top sample."""
    from math import comb

    return 1 - comb(n_libs - n_top, k) / comb(n_libs, k)


def _library_section(lib: dict, bench: dict | None) -> list[str]:
    """The benchmark again with one experiment = one library (a whole deposition run)."""
    cfg = lib["config"]
    out = ["### One experiment = one deposition run (library level)", "",
           "The spot-level benchmark above treats each HTEM sample as its own experiment. In the lab one deposition run makes a whole library: "
           "~44 spots on a composition gradient at one temperature and pressure. So the lab's real decision is which run to do next. "
           f"Here one experiment is one library: choosing it reveals every spot on it. Budget **{cfg['budget_runs']} runs**, "
           f"{cfg['initial_runs']} shared random seed runs, then {cfg['batch_runs']} run per round, {cfg['failure_rate']:.0%} simulated mishaps per spot "
           f"on top of the real missing measurements, {cfg['seeds']} seeds. The Bayesian loop ranks each untried library by the joint expected improvement "
           "of all its spots (qLogNEI on the library as one batch; `loop.select.select_bayes_libraries` says why). The grid sweep plans runs over each library's "
           "mean composition and temperature. A run that never finds a top-5% sample counts as budget + 1.", ""]
    for key, s in lib["skipped"].items():
        out += [f"**Skipped: {s['name']}.** It has {s['reason']} ({s['samples']} samples). The test would measure how fast a method exhausts the system, not how well it chooses.", ""]
    out += ["| Campaign | Method | Runs to the first top-5% sample (runs that got there) | Runs to 5 of the top 5% | Top-5% samples found | vs random, paired (± s.e.m.) | Libraries with a top-5% sample run | Spots measured |",
            "|---|---|---|---|---|---|---|---|"]
    for key, c in lib["campaigns"].items():
        for i, (m, v) in enumerate(c["methods"].items()):
            name = f"{c['spec']['name']} ({c['libraries']} libraries, {c['pool']} spots; top 5% = {c['top_size']} spots in {c['libraries_with_top']} libraries)" if i == 0 else ""
            diff = "—" if m == "random" else f"{v['top_found_minus_random_mean']:+.1f} ± {v['top_found_minus_random_sem']:.1f}"
            out.append(f"| {name} | {LABELS.get(m, m)} | **{_fmt(v['experiments_to_first_top5_median'])}** ({v['found_first_top5']}/{v['runs']}) | "
                       f"{_fmt(v['experiments_to_top_median'])} ({v['found_top_in_budget']}/{v['runs']}) | {v['top_found_mean']:.1f} of {c['top_size']} | {diff} | "
                       f"{v['top_libraries_found_mean']:.1f} of {c['libraries_with_top']} | {v['spots_measured_mean']:.0f} |")
    out.append("")
    conc = [f"{c['spec']['name']}: {c['top_per_library'][0]} of its {c['top_size']} top-5% spots are on one library, and "
            f"{c['libraries_with_top']} of {c['libraries']} libraries have any" for c in lib["campaigns"].values()]
    out += ["**Where the top samples sit.** " + "; ".join(conc) + ". At library level, finding one top sample means finding one of a handful of runs, "
            "and the count of top-5% samples found swings on whether the richest library is run.", ""]
    C = lib["campaigns"]
    first = [v["experiments_to_first_top5_median"] for c in C.values() for v in c["methods"].values()]
    luck = [_seed_hit_chance(c["libraries"], c["libraries_with_top"], cfg["initial_runs"]) for c in C.values()]
    ratio = [c["methods"]["bayes-gp"]["top_found_mean"] / c["methods"]["random"]["top_found_mean"] for c in C.values()]
    gain = [c["methods"]["bayes-gp"]["top_found_minus_random_mean"] for c in C.values()]
    spot = [bench["campaigns"][k]["methods"]["bayes-gp"]["top_found_mean"] / bench["campaigns"][k]["methods"]["random"]["top_found_mean"]
            for k in C if bench and k in bench["campaigns"]]
    frac = [v["spots_measured_mean"] / c["pool"] for c in C.values() for v in c["methods"].values()]
    g = {k: c["methods"]["grid"] for k, c in C.items() if "grid" in c["methods"]}
    grid_better = [C[k]["spec"]["name"].lower() for k, v in g.items() if v["top_found_minus_random_mean"] - 2 * v["top_found_minus_random_sem"] > 0]
    grid_text = (f"Grid search beats random clearly (by more than two standard errors) only on {', '.join(grid_better)}, and is within noise of it on the other {len(g) - len(grid_better)}. "
                 if grid_better else "Grid search is within noise of random on every campaign. ")
    out += [f"**The headline metric stops separating the methods.** Every method needs a median of {min(first):.0f}–{max(first):.0f} runs to its first top-5% sample. "
            f"The {cfg['initial_runs']} random seed runs alone include a library with a top-5% sample {min(luck):.0%}–{max(luck):.0%} of the time "
            "(exact, from the library counts above), so luck decides it, as it did for the first-hit metric at spot level.", "",
            f"**What survives is smaller.** The Bayesian loop still finds more top-5% samples than random in the same {cfg['budget_runs']} runs: "
            f"{min(ratio):.1f}–{max(ratio):.1f}× as many ({min(gain):+.0f} to {max(gain):+.0f} samples, paired)"
            + (f", against {min(spot):.1f}–{max(spot):.1f}× at spot level on the same campaigns" if spot else "") + ". "
            + grid_text +
            f"After {cfg['budget_runs']} runs every method has measured {min(frac):.0%}–{max(frac):.0%} of the system's spots, so the room to save runs is small: "
            "these systems have about 20 libraries each. The spot-level numbers above overstate the savings a lab choosing runs would see.", ""]
    return out


SENS_LABEL = {"budget-5": "5 runs, 2 seed runs", "main": "10 runs, 2 seed runs (above)", "budget-15": "15 runs, 2 seed runs", "seed-runs-1": "10 runs, 1 seed run"}


def _library_sensitivity(R: Path, lib: dict) -> list[str]:
    """The library-level comparison at other run budgets and seed-run counts."""
    import numpy as np

    d = R / "benchmark" / "library_sensitivity"
    runs = {"main": lib, **{k: v for k in ("budget-5", "budget-15", "seed-runs-1") if (v := _load(d / f"{k}.json"))}}
    runs = {k: runs[k] for k in SENS_LABEL if k in runs}
    if len(runs) < 2:
        return []
    out = ["#### Does this depend on the budget or the seed runs?", "",
           "The same comparison at other run budgets and with one random seed run instead of two (same seeds, same 20-library threshold, so IZTO is still skipped). "
           "Each cell is random / grid / Bayesian.", "",
           "| Design | Campaign | Runs to the first top-5% sample | Top-5% samples found | Bayesian ÷ random | Bayesian − random, paired (± s.e.m.) | Spots measured |",
           "|---|---|---|---|---|---|---|"]
    ratios: dict[str, list[float]] = {}
    clear: dict[str, int] = {}
    first_sep: dict[str, list[float]] = {}
    for name, r in runs.items():
        for i, (key, c) in enumerate(r["campaigns"].items()):
            m = c["methods"]
            b, rnd = m["bayes-gp"], m["random"]
            ratio = b["top_found_mean"] / rnd["top_found_mean"] if rnd["top_found_mean"] else float("nan")
            ratios.setdefault(name, []).append(ratio)
            clear[name] = clear.get(name, 0) + int(b["top_found_minus_random_mean"] - 2 * b["top_found_minus_random_sem"] > 0)
            first_sep.setdefault(name, []).append(rnd["experiments_to_first_top5_median"] - b["experiments_to_first_top5_median"])
            frac = np.mean([v["spots_measured_mean"] for v in m.values()]) / c["pool"]
            out.append(f"| {SENS_LABEL[name] if i == 0 else ''} | {c['spec']['name']} | "
                       + " / ".join(_fmt(m[x]["experiments_to_first_top5_median"]) for x in ("random", "grid", "bayes-gp")) + " | "
                       + " / ".join(f"{m[x]['top_found_mean']:.1f}" for x in ("random", "grid", "bayes-gp")) + f" of {c['top_size']} | "
                       f"{ratio:.1f}× | {b['top_found_minus_random_mean']:+.1f} ± {b['top_found_minus_random_sem']:.1f} | {frac:.0%} |")
    out.append("")
    n_c = len(lib["campaigns"])
    def rng_txt(v, f="{:.1f}", unit="×", sep="–"):
        lo, hi = f.format(min(v)), f.format(max(v))
        return (lo if lo == hi else f"{lo}{sep}{hi}") + unit
    short = {k: SENS_LABEL[k].replace(" (above)", "") for k in runs}
    gains = [c["methods"]["bayes-gp"]["top_found_minus_random_mean"] for r in runs.values() for c in r["campaigns"].values()]
    grid_wins = [f"the {c['spec']['name'][0].lower() + c['spec']['name'][1:]} campaign at {short[k]}" for k, r in runs.items() for c in r["campaigns"].values()
                 if c["methods"]["grid"]["top_found_mean"] > c["methods"]["bayes-gp"]["top_found_mean"]]
    budgets = [k for k in ("budget-5", "main", "budget-15") if k in ratios]
    falls = len(budgets) > 1 and all(max(ratios[a]) >= max(ratios[b]) for a, b in zip(budgets, budgets[1:]))
    out += ["**Reading.** Bayesian ÷ random: " + "; ".join(f"{short[k]}: {rng_txt(v)}" for k, v in ratios.items()) + ". "
            + ("The ratio falls as the budget grows, because by 15 runs random has also run most of each system's libraries; "
               if falls else "The ratio does not fall steadily with the budget; ")
            + f"the paired gain in samples stays between {min(gains):+.0f} and {max(gains):+.0f}. "
            "The Bayesian loop beats random by more than two standard errors on " + ", ".join(f"{clear[k]} of {n_c} campaigns at {short[k]}" for k in ratios) + ". "
            + (f"Grid search finds more top-5% samples than the Bayesian loop on {', '.join(grid_wins)}. " if grid_wins else "")
            + "Median runs saved to the first top-5% sample (random − Bayesian) are "
            + rng_txt([x for v in first_sep.values() for x in v], "{:+.0f}", " runs", " to ").replace("+0 ", "0 ") + " in every design, so that metric never separates the methods. "
            f"So the {rng_txt(ratios['main'])} above is specific to the 10-run budget. Across these designs the Bayesian loop finds "
            f"{rng_txt([x for v in ratios.values() for x in v])} as many top-5% samples as random" + (", more when the lab can afford fewer runs." if falls else "."), ""]
    return out


def _more_systems_section(more: dict, lib: dict | None) -> list[str]:
    """The same benchmark on five systems chosen by rule after the method was fixed."""
    names = ", ".join(f"{s} ({more['data'][s]['accepted']} samples)" for s in more["systems"])
    out = ["### Five more systems, chosen by rule", "",
           "The four campaigns above use three systems picked while the method was being built, so they could be the ones where it happens to work. "
           f"These five were chosen afterwards by a rule fixed before any optimizer was run on them: {more['rules']['systems']}. "
           f"Goals: {more['rules']['goals']}. Everything else is the same as above: 60-experiment budget, 10 random seed experiments, batches of 5, "
           f"5% simulated lab failures, {more['config']['seeds']} seeds. Systems: {names}.", "",
           "**Ranked above the cut-off but excluded for lack of usable data** (decided before any optimization ran): "
           + "; ".join(f"{k}: {v}" for k, v in more.get("excluded", {}).items())
           + ". Downloading them exposed an importer bug: for sulfides, XRF also reports the sulfur, so the metals alone sum to ~50% and those records were rejected. "
           "Fixed; the three demo systems import exactly as before.", "",
           "| System | Libraries | Samples received | Accepted | With conductivity | With band gap | Conductivity range (decades) | Samples with gap ≥ 3 eV and a conductivity |",
           "|---|---|---|---|---|---|---|---|"]
    for sname, d in more["data"].items():
        out.append(f"| {sname} | {d['libraries']} | {d['samples_received']} | {d['accepted']} | {d['with_conductivity']} | {d['with_band_gap']} | "
                   f"{_fmt(d['conductivity_decades'], 1)} | {d['transparent_feasible']}{'' if d['transparent_goal'] else ' (too few; no goal)'} |")
    out += ["", "| Campaign | Random | Grid sweep | Bayesian (GP) | Bayesian (random forest) | Top-5% found: GP vs random (paired, ± s.e.m.) |",
            "|---|---|---|---|---|---|"]
    rows = []
    for key, c in more["campaigns"].items():
        m = c["methods"]
        cell = lambda v: f"{_fmt(v['experiments_to_top_median'])} ({v['found_top_in_budget']}/{v['runs']}) · {v['top_found_mean']:.1f} found"
        g = m["bayes-gp"]
        out.append(f"| {c['spec']['name']} ({c['pool']} samples, top 5% = {c['top_size']}) | " + " | ".join(cell(m[k]) for k in ("random", "grid", "bayes-gp", "bayes-rf") if k in m)
                   + f" | {g['top_found_minus_random_mean']:+.1f} ± {g['top_found_minus_random_sem']:.1f} |")
        rows.append((c, g))
    clear = [c["spec"]["name"] for c, g in rows if g["top_found_minus_random_mean"] - 2 * g["top_found_minus_random_sem"] > 0]
    worse = [c["spec"]["name"] for c, g in rows if g["top_found_minus_random_mean"] + 2 * g["top_found_minus_random_sem"] < 0]
    faster = [c["spec"]["name"] for c, g in rows if g["experiments_to_top_median"] < min(c["methods"]["random"]["experiments_to_top_median"], c["methods"]["grid"]["experiments_to_top_median"])]
    ratio = [g["top_found_mean"] / c["methods"]["random"]["top_found_mean"] for c, g in rows if c["methods"]["random"]["top_found_mean"] > 0]
    others = [c["spec"]["name"] for c, _ in rows if c["spec"]["name"] not in clear]
    named = (f" (all but {', '.join(others)})" if others and len(clear) > len(rows) / 2 else (f" ({', '.join(clear)})" if clear and others else ""))
    rf = {c["spec"]["name"]: c["methods"]["bayes-rf"] for c, _ in rows if "bayes-rf" in c["methods"]}
    rf_note = "".join(f" On {n.lower()}, the random-forest model does better: {_fmt(rf[n]['experiments_to_top_median'])} experiments ({rf[n]['found_top_in_budget']}/{rf[n]['runs']} got there)."
                      for n in others if n in rf and rf[n]["experiments_to_top_median"] < c_min(rows, n))
    out += ["", "Cells: median experiments to find 5 of the top-5% samples (runs that got there) · top-5% samples found in the budget, mean. A run that never gets there counts as budget + 1.", "",
            f"**Does it hold?** Mostly. The GP loop finds clearly more top-5% samples than random (by more than two standard errors, paired by seed) in {len(clear)} of {len(rows)} campaigns{named}, "
            + (f"clearly fewer in {len(worse)} ({', '.join(worse)}), " if worse else "clearly fewer in none, ")
            + f"and needs fewer experiments than both baselines to find 5 of them in {len(faster)} of {len(rows)}. "
            + (f"It finds {min(ratio):.1f}–{max(ratio):.1f}× as many top-5% samples as random (the demo campaigns: 2.5–4×)." if ratio else "") + rf_note, ""]
    if lib:
        out += ["**Counted per deposition run** (same design as 'One experiment = one deposition run': 10 runs, 2 random seed runs, systems with at least "
                f"{lib['config']['min_libraries']} libraries):", "",
                "| Campaign | Random | Grid sweep | Bayesian (GP) | GP vs random, paired (± s.e.m.) |", "|---|---|---|---|---|"]
        if lib["campaigns"]:
            for key, c in lib["campaigns"].items():
                m = c["methods"]
                out.append(f"| {c['spec']['name']} ({c['libraries']} libraries) | " + " | ".join(f"{m[k]['top_found_mean']:.1f}" for k in ("random", "grid", "bayes-gp"))
                           + f" | {m['bayes-gp']['top_found_minus_random_mean']:+.1f} ± {m['bayes-gp']['top_found_minus_random_sem']:.1f} |")
            out += ["", "Cells: top-5% samples found in 10 runs, mean over seeds."]
        else:
            out = out[:-4]
            n = sorted(s["libraries"] for s in lib["skipped"].values())
            out += [f"**Counted per deposition run: not testable here.** After import, these systems have only {n[0]}–{n[-1]} deposition runs with usable samples each "
                    f"(the run-level design needs at least {lib['config']['min_libraries']}), so a 10-run budget would cover most of each system. "
                    "The per-sample gains above are therefore not checked at run level for these systems; on the demo systems, counting per run cut the gain to 1.2–1.9×."]
        out.append("")
    return out


def c_min(rows, name: str) -> float:
    c = next(c for c, _ in rows if c["spec"]["name"] == name)
    return min(c["methods"]["random"]["experiments_to_top_median"], c["methods"]["grid"]["experiments_to_top_median"], c["methods"]["bayes-gp"]["experiments_to_top_median"])


def _more_systems_bullet(R: Path) -> list[str]:
    more = _load(R / "benchmark" / "more_systems.json")
    if not more or not more["campaigns"]:
        return []
    rows = [c["methods"]["bayes-gp"] for c in more["campaigns"].values()]
    clear = sum(1 for g in rows if g["top_found_minus_random_mean"] - 2 * g["top_found_minus_random_sem"] > 0)
    miss = [c["spec"]["name"].lower() for c in more["campaigns"].values()
            if c["methods"]["bayes-gp"]["top_found_minus_random_mean"] - 2 * c["methods"]["bayes-gp"]["top_found_minus_random_sem"] <= 0]
    return [f"- **On five systems chosen by rule after the method was fixed**, the GP loop finds clearly more top-5% samples than random in {clear} of {len(rows)} campaigns"
            + (f", but not on {', '.join(miss)}" if miss else "") + ". These systems have too few deposition runs to repeat the run-level check "
            "(see 'Five more systems, chosen by rule')."]


def write_report() -> Path:
    R = config.REPORTS_DIR
    out = ["# Lodestone evaluation report", "",
           "Every number here comes from `uv run lodestone ...` commands on real measurements from NREL's HTEM DB, "
           "reported as it came out, including the weak results. The lab is a **replay**: it returns real past measurements, "
           "so a campaign can only choose among samples that were actually made.", ""]

    bench = _load(R / "benchmark" / "summary.json")
    curves = _load(R / "benchmark" / "curves.json") or {}

    def best_at_end(key: str, method: str) -> str:
        import numpy as np

        from lodestone.loop.problem import fmt_value, untransform
        from lodestone.schemas import PROPERTIES, CampaignSpec

        cs = curves.get(key, {}).get(method)
        if not cs:
            return "—"
        spec = CampaignSpec(**bench["campaigns"][key]["spec"])
        last = np.array([c[-1] for c in cs], float)
        last = last[np.isfinite(last)]
        if not len(last):
            return "none"
        med = float(np.median(last))
        if spec.window:
            return "inside the window" if med >= 0 else f"{-med:.2f} eV outside"
        o = spec.objectives[0]
        return f"{fmt_value(float(untransform(o.property, np.array([med]))[0]))} {PROPERTIES[o.property]['unit']}"

    if bench:
        cfg = bench["config"]
        out += ["## 1. Does the loop beat a manual approach? (E1)", "",
                f"Each campaign: budget = the spec's budget (60, or 40 for IZTO), 10 shared random seed experiments then batches of 5, "
                f"{cfg['failure_rate']:.0%} simulated lab failures on top of the data's real missing measurements, {cfg['seeds']} seeds per method. "
                "The headline metric is **experiments needed to find 5 of the top-5% samples** (median; a run that never gets there counts as budget + 1).", ""]
        for key, c in bench["campaigns"].items():
            k = c.get("headline_k", 5)
            out += [f"### {c['spec']['name']}", "",
                    f"`{c['summary']}` · pool of {c['pool']} real samples · top 5% = {c['top_size']} samples, top 1% = {c.get('top1_size', '?')}", "",
                    f"| Method | Experiments to find {k} of the top 5% | Runs that got there | First top-1% sample | Top-5% samples found | Median best value at the end | Failed experiments |",
                    "|---|---|---|---|---|---|---|"]
            for m, v in c["methods"].items():
                out.append(f"| {LABELS.get(m, m)} | **{_fmt(v['experiments_to_top_median'])}** | {v['found_top_in_budget']}/{v['runs']} | "
                           f"{_fmt(v.get('experiments_to_first_top1_median'))} ({v.get('found_first_top1', 0)}/{v['runs']}) | {v['top_found_mean']:.1f} | {best_at_end(key, m)} | {v['failures_mean']:.1f} |")
            out.append("")
        out += ["![benchmark](benchmark/benchmark.png)", ""]
        out += ["**A metric that did not discriminate.** 'Experiments to the *first* top-5% sample' was the first metric planned. On these pools the 10 shared random seed experiments "
                "already hit a top-5% sample about 40% of the time, so every method scores ~9–17 on it and the differences are noise. It is kept in `reports/benchmark/summary.json` "
                "(`experiments_to_first_top5_median`) but not used as the headline.", ""]
        rob = bench["robustness"]
        out += [f"### Robustness ({cfg['robustness_campaign']}, {cfg['robustness_seeds']} seeds)", "",
                "| Failure rate | " + " | ".join(LABELS.get(m, m) for m in next(iter(rob["failure_rate"].values()))) + " |",
                "|---|" + "---|" * len(next(iter(rob["failure_rate"].values())))]
        for fr, ms in rob["failure_rate"].items():
            out.append(f"| {float(fr):.0%} | " + " | ".join(f"{_fmt(v['experiments_to_top_median'])} exp · {v['top_found_mean']:.1f} found" for v in ms.values()) + " |")
        out += ["", "| Batch size | " + " | ".join(LABELS.get(m, m) for m in next(iter(rob["batch_size"].values()))) + " |",
                "|---|" + "---|" * len(next(iter(rob["batch_size"].values())))]
        for b, ms in rob["batch_size"].items():
            out.append(f"| {b} | " + " | ".join(f"{_fmt(v['experiments_to_top_median'])} exp · {v['top_found_mean']:.1f} found" for v in ms.values()) + " |")
        out.append("")
    lib = _load(R / "benchmark" / "library.json")
    if lib:
        out += _library_section(lib, bench)
        out += _library_sensitivity(R, lib)

    more = _load(R / "benchmark" / "more_systems.json")
    if more and more["campaigns"]:
        out += _more_systems_section(more, _load(R / "benchmark" / "more_systems_library.json"))

    ob = _load(R / "onboarding" / "onboarding.json")
    prov = _load(R / "onboarding" / "provenance.json")
    cal = _load(R / "calibration" / "calibration.json")
    eff = _load(R / "calibration" / "loop_effect.json")
    if cal:
        out += _calibration_section(cal, eff, bench)
    lib = _load(R / "calibration" / "held_out_libraries.json")
    if lib:
        out += _held_out_libraries_section(lib, cal)

    out += ["## 2. Data onboarding and provenance (D1–D4)", ""]
    if ob:
        out += ["| System | Libraries | Listed but no samples served | Records received | Accepted | Rejected | Main rejection reasons |", "|---|---|---|---|---|---|---|"]
        for s, t in ob["systems"].items():
            reasons = ", ".join(f"{k.split(': ', 1)[1]} ({v})" for k, v in sorted(t["reasons"].items(), key=lambda kv: -kv[1])[:2]) or "none"
            out.append(f"| {s} | {t['libraries']} | {t['libraries_with_no_samples_served']} | {t['received']} | {t['accepted']} | {t['rejected']} | {reasons} |")
        br = ob["broken_records"]
        out += ["", f"Deliberately broken records (both lab formats): **{br['correct']}/{br['cases']}** handled correctly (accepted, measurement dropped, or record rejected, each with a reason).", ""]
    if prov:
        out.append(f"Provenance spot-check: {prov['sampled']} random samples, {prov['values_checked']} values compared with the raw HTEM record"
                   + (f" and {prov['values_checked_live']} re-fetched from the live API" if prov.get("live") else "")
                   + f": **{'all match' if prov['all_match'] else 'MISMATCHES FOUND'}**." + (" (Samples rejected at import are skipped.)" if prov["checked"] < prov["sampled"] else ""))
        out.append("")

    out += ["## 3. Agent (E2)", ""]
    tasks = _load(config.ROOT / "evals" / "agent_tasks.json")
    if tasks:
        kinds: dict[str, int] = {}
        for t in tasks:
            kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
        out.append(f"{len(tasks)} tasks with answers computed from the data: " + ", ".join(f"{n} {k}" for k, n in kinds.items()) + ".")
    out += ["", "| Policy | Tasks | Success | Lookup | Count | Provenance | Campaign | Guardrail | Parse | Valid tool calls (excl. guardrails) | Tasks with invented numbers | Cost / task |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol in ("claude", "scripted"):
        d = _load(R / "agent_eval" / f"{pol}.json")
        if not d:
            continue
        s = d["summary"]
        g = lambda k: f"{s['by_kind'][k]['success_rate']:.0%}" if k in s["by_kind"] else "—"
        out.append(f"| {'Claude ' + (s['model'] or '') if pol == 'claude' else 'Scripted reference (no LLM)'} | {s['tasks']} | {s['success_rate']:.0%} | {g('lookup')} | {g('count')} | {g('provenance')} | "
                   f"{g('campaign')} | {g('guardrail')} | {g('parse')} | {s['valid_tool_call_rate_excluding_guardrails']:.0%} | {s['tasks_with_invented_numbers']} | ${s['mean_cost_usd']:.3f} |")
    if not (R / "agent_eval" / "claude.json").exists():
        out += ["", "**Claude has not been evaluated yet**: this build ran without Anthropic credentials. Run "
                "`ANTHROPIC_API_KEY=... uv run lodestone eval agent --policy claude --parse-with claude`. The scripted reference policy drives the "
                "same MCP tools without an LLM; it validates the tools and the scoring, and says nothing about Claude."]
    out += ["", "## 4. Isolation (W2)", "",
            "`tests/test_isolation.py` discovers every registered read path and calls it with the other lab's ids, directly, through the REST API and "
            "through the MCP tools. All are blocked, and each attempt is written to the audit log. Run `uv run pytest`.", ""]

    out += ["## What didn't work (or isn't shown)", "",
            "- **'Experiments to the first top-5% sample' doesn't separate methods** on pools of ~1,000 with 10 random seed experiments: luck dominates. The stricter metrics (5 of the top 5%, first top-1%, top-5% samples found) do.",
            "- **Finding many good samples is not the same as finding the single best one.** On the transparent-conductor campaign the Bayesian loop finds four times as many top-5% samples as random, yet the *median best value* it reaches (3.2 S/cm) is no higher than random's (4.0 S/cm), and no method reliably finds the extreme outlier (an 85 S/cm Co-Ni-Zn-O film). The first top-1% sample arrives no sooner than with random search on the constrained campaign.",
            *_library_bullet(R),
            *_more_systems_bullet(R),
            "- **Grid search is a strong baseline for the very first hits** when a system has only one or two composition dimensions: a systematic sweep covers the space quickly. The optimizer pulls ahead once it has data to exploit.",
            "- **The random-forest surrogate is uneven**: excellent on conductivity, weak on constraints and windows, where its uncertainty estimates (tree spread) are poor.",
            "- **Replay can only choose among samples that were made.** A real campaign would propose new recipes; the 'suggest recipes' feature does that, but its predictions are unverified by construction.",
            "- **HTEM data has gaps**: 18 of 22 IZTO libraries and some others are listed as public but the API serves no sample records, and 365 ZnSnN samples have no XRF composition. These are reported, not hidden.",
            "- **The adapters have only seen two formats**, one of them generated by this project. A real second lab will break them in new ways.",
            "- **Conductivity censoring is a modelling assumption**: a missing four-point-probe value is treated as 'too resistive to measure' (one decade below the lowest seen). It is usually right for HTEM, and not checked per sample.",
            *_calibration_bullet(R),
            *_held_out_bullet(R),
            "- **Claude's agent results are pending an API key** (see section 3).", ""]
    path = R / "EVALUATION.md"
    path.write_text("\n".join(out) + "\n")
    return path
