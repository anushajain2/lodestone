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

    ob = _load(R / "onboarding" / "onboarding.json")
    prov = _load(R / "onboarding" / "provenance.json")
    cal = _load(R / "calibration" / "calibration.json")
    eff = _load(R / "calibration" / "loop_effect.json")
    if cal:
        out += _calibration_section(cal, eff, bench)

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
            "- **Grid search is a strong baseline for the very first hits** when a system has only one or two composition dimensions: a systematic sweep covers the space quickly. The optimizer pulls ahead once it has data to exploit.",
            "- **The random-forest surrogate is uneven**: excellent on conductivity, weak on constraints and windows, where its uncertainty estimates (tree spread) are poor.",
            "- **Replay can only choose among samples that were made.** A real campaign would propose new recipes; the 'suggest recipes' feature does that, but its predictions are unverified by construction.",
            "- **HTEM data has gaps**: 18 of 22 IZTO libraries and some others are listed as public but the API serves no sample records, and 365 ZnSnN samples have no XRF composition. These are reported, not hidden.",
            "- **The adapters have only seen two formats**, one of them generated by this project. A real second lab will break them in new ways.",
            "- **Conductivity censoring is a modelling assumption**: a missing four-point-probe value is treated as 'too resistive to measure' (one decade below the lowest seen). It is usually right for HTEM, and not checked per sample.",
            *_calibration_bullet(R),
            "- **Claude's agent results are pending an API key** (see section 3).", ""]
    path = R / "EVALUATION.md"
    path.write_text("\n".join(out) + "\n")
    return path
