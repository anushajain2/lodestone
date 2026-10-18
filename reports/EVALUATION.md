# Lodestone evaluation report

Every number here comes from `uv run lodestone ...` commands on real measurements from NREL's HTEM DB, reported as it came out, including the weak results. The lab is a **replay**: it returns real past measurements, so a campaign can only choose among samples that were actually made.

## 1. Does the loop beat a manual approach? (E1)

Each campaign: budget = the spec's budget (60, or 40 for IZTO), 10 shared random seed experiments then batches of 5, 5% simulated lab failures on top of the data's real missing measurements, 20 seeds per method. The headline metric is **experiments needed to find 5 of the top-5% samples** (median; a run that never gets there counts as budget + 1).

### Most conductive zinc tin nitride

`N-Sn-Zn: maximize conductivity; budget 60, 5/round, bayes` · pool of 999 real samples · top 5% = 50 samples, top 1% = 10

| Method | Experiments to find 5 of the top 5% | Runs that got there | First top-1% sample | Top-5% samples found | Median best value at the end | Failed experiments |
|---|---|---|---|---|---|---|
| Random | **61** | 6/20 | 46 (14/20) | 3.5 | 7,594 S/cm | 7.2 |
| Grid sweep | **61** | 3/20 | 57 (10/20) | 2.6 | 6,300 S/cm | 6.0 |
| Bayesian (GP) | **39** | 19/20 | 14 (20/20) | 9.4 | 9,348 S/cm | 6.3 |
| Bayesian (random forest) | **30** | 20/20 | 21 (20/20) | 13.2 | 9,374 S/cm | 8.7 |

### Transparent p-type conductor (gap ≥ 3 eV)

`Co-Ni-O-Zn: maximize conductivity with direct band gap >= 3 eV; budget 60, 5/round, bayes` · pool of 880 real samples · top 5% = 44 samples, top 1% = 9

| Method | Experiments to find 5 of the top 5% | Runs that got there | First top-1% sample | Top-5% samples found | Median best value at the end | Failed experiments |
|---|---|---|---|---|---|---|
| Random | **61** | 3/20 | 52 (11/20) | 2.8 | 4.01 S/cm | 3.2 |
| Grid sweep | **56** | 11/20 | 58 (10/20) | 4.7 | 3.41 S/cm | 2.9 |
| Bayesian (GP) | **31** | 18/20 | 52 (10/20) | 11.9 | 3.24 S/cm | 3.3 |
| Bayesian (random forest) | **37** | 15/20 | 61 (7/20) | 12.5 | 2.93 S/cm | 3.1 |

### Band gap in the 1.4–1.6 eV solar window

`N-Sn-Zn: direct band gap in 1.4–1.6 eV; budget 60, 5/round, bayes` · pool of 999 real samples · top 5% = 50 samples, top 1% = 17

| Method | Experiments to find 5 of the top 5% | Runs that got there | First top-1% sample | Top-5% samples found | Median best value at the end | Failed experiments |
|---|---|---|---|---|---|---|
| Random | **61** | 5/20 | 61 (9/20) | 2.8 | 0.02 eV outside | 3.1 |
| Grid sweep | **61** | 3/20 | 58 (10/20) | 2.6 | 0.00 eV outside | 3.4 |
| Bayesian (GP) | **50** | 16/20 | 54 (12/20) | 7.2 | inside the window | 3.1 |
| Bayesian (random forest) | **61** | 5/20 | 46 (13/20) | 3.0 | inside the window | 3.5 |

### Most conductive indium zinc tin oxide

`In-O-Sn-Zn: maximize conductivity; budget 40, 5/round, bayes` · pool of 176 real samples · top 5% = 9 samples, top 1% = 2

| Method | Experiments to find 5 of the top 5% | Runs that got there | First top-1% sample | Top-5% samples found | Median best value at the end | Failed experiments |
|---|---|---|---|---|---|---|
| Random | **41** | 0/20 | 41 (6/20) | 1.4 | 1,345 S/cm | 2.1 |
| Grid sweep | **41** | 0/20 | 41 (7/20) | 1.6 | 1,345 S/cm | 2.2 |
| Bayesian (GP) | **32** | 16/20 | 24 (17/20) | 5.3 | 1,454 S/cm | 2.6 |
| Bayesian (random forest) | **30** | 17/20 | 18 (19/20) | 5.9 | 1,454 S/cm | 2.2 |

![benchmark](benchmark/benchmark.png)

**A metric that did not discriminate.** 'Experiments to the *first* top-5% sample' was the first metric planned. On these pools the 10 shared random seed experiments already hit a top-5% sample about 40% of the time, so every method scores ~9–17 on it and the differences are noise. It is kept in `reports/benchmark/summary.json` (`experiments_to_first_top5_median`) but not used as the headline.

### Robustness (cnzo-transparent-conductor, 10 seeds)

| Failure rate | Random | Grid sweep | Bayesian (GP) |
|---|---|---|---|
| 0% | 61 exp · 3.2 found | 60 exp · 4.7 found | 34 exp · 13.0 found |
| 10% | 61 exp · 2.9 found | 61 exp · 3.9 found | 30 exp · 10.9 found |
| 20% | 61 exp · 2.4 found | 61 exp · 3.5 found | 36 exp · 10.5 found |

| Batch size | Random | Bayesian (GP) |
|---|---|---|
| 1 | 61 exp · 2.6 found | 28 exp · 13.9 found |
| 5 | 61 exp · 3.3 found | 32 exp · 11.9 found |
| 10 | 61 exp · 2.7 found | 32 exp · 10.4 found |

### Is the model's uncertainty honest?

Fit the GP on n random measured samples of a system, predict every other measured sample, and count how often the real value falls inside the predicted 68% and 95% intervals (20 random splits each). A calibrated model scores 68% and 95%. Two fixes were tried. **CV variance scaling** widens the intervals by a factor from 5-fold cross-validation on the campaign's own data. **Fixed inflation** widens each property's intervals by a constant learned only from the *other* systems' held-out checks. A noise floor and a Matérn-5/2 kernel were also tried; neither raised 68% coverage meaningfully (it moved by at most 7 percentage points, in either direction; `reports/calibration/calibration.json`). Neither fix changes the predicted values, so R² is the same for all three.

| System | Property | n | 68% interval, default | CV scaling | Fixed inflation | 95% interval, default | CV scaling | Fixed inflation | R² |
|---|---|---|---|---|---|---|---|---|---|
| Co-Ni-O-Zn | band gap | 30 | 46% | 80% | 68% | 67% | 92% | 86% | 0.07 |
| Co-Ni-O-Zn | band gap | 60 | 43% | 88% | 66% | 65% | 97% | 85% | 0.19 |
| Co-Ni-O-Zn | log10 conductivity | 30 | 55% | 91% | 73% | 79% | 98% | 91% | 0.30 |
| Co-Ni-O-Zn | log10 conductivity | 60 | 61% | 94% | 78% | 84% | 98% | 93% | 0.51 |
| In-O-Sn-Zn | log10 conductivity | 30 | 50% | 90% | 64% | 74% | 98% | 86% | 0.65 |
| In-O-Sn-Zn | log10 conductivity | 60 | 49% | 93% | 63% | 72% | 99% | 84% | 0.74 |
| N-Sn-Zn | band gap | 30 | 43% | 85% | 69% | 66% | 95% | 87% | 0.28 |
| N-Sn-Zn | band gap | 60 | 45% | 93% | 72% | 69% | 99% | 90% | 0.34 |
| N-Sn-Zn | log10 conductivity | 30 | 52% | 84% | 68% | 75% | 95% | 87% | 0.63 |
| N-Sn-Zn | log10 conductivity | 60 | 55% | 87% | 71% | 77% | 96% | 88% | 0.75 |

The default GP is overconfident everywhere. CV scaling overshoots: it widens the intervals 3.0–5.0× and lands near 85–95% coverage. Fixed inflation (1.48–2.13×) lands close to nominal.

**Does honest uncertainty help the loop? No.** The same Bayesian (GP) campaigns, same seeds and 5% failure rate, paired seed by seed with the default (20 seeds; the default rerun matches section 1 exactly):

| Campaign | Default: experiments to 5 of top 5% (runs that got there) · top-5% found | CV scaling | Fixed inflation | Top-5% found, fixed − default (paired, ± s.e.m.) |
|---|---|---|---|---|
| Most conductive zinc tin nitride | 39 (19/20) · 9.4 | 50 (15/20) · 5.5 | 40 (19/20) · 7.5 | -1.9 ± 0.7 |
| Transparent p-type conductor (gap ≥ 3 eV) | 31 (18/20) · 11.9 | 58 (12/20) · 5.3 | 40 (15/20) · 7.0 | -5.0 ± 1.0 |
| Band gap in the 1.4–1.6 eV solar window | 50 (16/20) · 7.2 | 55 (11/20) · 5.2 | 44 (16/20) · 7.0 | -0.2 ± 1.0 |
| Most conductive indium zinc tin oxide | 32 (16/20) · 5.3 | 41 (3/20) · 3.4 | 36 (13/20) · 4.7 | -0.7 ± 0.5 |

So the shipped GP keeps its default, overconfident uncertainty (`models.GP_OPTIONS`; `uv run lodestone eval calibration` reruns all of this). A likely reading, not tested: with ~1,000 samples where the good ones cluster, an overconfident model exploits sooner, and that pays. The raw GP's ± is too narrow by about ×1.6 for log10 conductivity and ×2.1 for band gap. So the uncertainty *shown to people* (the ± and the chance of meeting a constraint in 'why this sample', the dashboard and the agent tools) uses the fixed inflation for the campaign's system, labelled as calibrated on held-out data, while the optimizer ranks with the raw GP (`loop.calibration.display_inflation`; a test checks that a seeded run picks the same samples with and without it).

## 2. Data onboarding and provenance (D1–D4)

| System | Libraries | Listed but no samples served | Records received | Accepted | Rejected | Main rejection reasons |
|---|---|---|---|---|---|---|
| Co-Ni-O-Zn | 24 | 4 | 880 | 880 | 0 | library lists 44 samples but the API served no sample records (4) |
| In-O-Sn-Zn | 22 | 18 | 176 | 176 | 0 | library lists 44 samples but the API served no sample records (18), deposition temperature not recorded for this library (4) |
| N-Sn-Zn | 33 | 2 | 1364 | 999 | 365 | no usable XRF composition (352), unreadable XRF entry 'Zn'=None (13) |

Deliberately broken records (both lab formats): **18/18** handled correctly (accepted, measurement dropped, or record rejected, each with a reason).

Provenance spot-check: 20 random samples, 56 values compared with the raw HTEM record and 56 re-fetched from the live API: **all match**. (Samples rejected at import are skipped.)

## 3. Agent (E2)

30 tasks with answers computed from the data: 8 lookup, 2 count, 3 provenance, 5 campaign, 6 guardrail, 6 parse.

| Policy | Tasks | Success | Lookup | Count | Provenance | Campaign | Guardrail | Parse | Valid tool calls (excl. guardrails) | Tasks with invented numbers | Cost / task |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Scripted reference (no LLM) | 30 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0 | $0.000 |

**Claude has not been evaluated yet**: this build ran without Anthropic credentials. Run `ANTHROPIC_API_KEY=... uv run lodestone eval agent --policy claude --parse-with claude`. The scripted reference policy drives the same MCP tools without an LLM; it validates the tools and the scoring, and says nothing about Claude.

## 4. Isolation (W2)

`tests/test_isolation.py` discovers every registered read path and calls it with the other lab's ids, directly, through the REST API and through the MCP tools. All are blocked, and each attempt is written to the audit log. Run `uv run pytest`.

## What didn't work (or isn't shown)

- **'Experiments to the first top-5% sample' doesn't separate methods** on pools of ~1,000 with 10 random seed experiments: luck dominates. The stricter metrics (5 of the top 5%, first top-1%, top-5% samples found) do.
- **Finding many good samples is not the same as finding the single best one.** On the transparent-conductor campaign the Bayesian loop finds four times as many top-5% samples as random, yet the *median best value* it reaches (3.2 S/cm) is no higher than random's (4.0 S/cm), and no method reliably finds the extreme outlier (an 85 S/cm Co-Ni-Zn-O film). The first top-1% sample arrives no sooner than with random search on the constrained campaign.
- **Grid search is a strong baseline for the very first hits** when a system has only one or two composition dimensions: a systematic sweep covers the space quickly. The optimizer pulls ahead once it has data to exploit.
- **The random-forest surrogate is uneven**: excellent on conductivity, weak on constraints and windows, where its uncertainty estimates (tree spread) are poor.
- **Replay can only choose among samples that were made.** A real campaign would propose new recipes; the 'suggest recipes' feature does that, but its predictions are unverified by construction.
- **HTEM data has gaps**: 18 of 22 IZTO libraries and some others are listed as public but the API serves no sample records, and 365 ZnSnN samples have no XRF composition. These are reported, not hidden.
- **The adapters have only seen two formats**, one of them generated by this project. A real second lab will break them in new ways.
- **Conductivity censoring is a modelling assumption**: a missing four-point-probe value is treated as 'too resistive to measure' (one decade below the lowest seen). It is usually right for HTEM, and not checked per sample.
- **The GP's error bars are too narrow, and fixing them made the loop worse.** Its 68% intervals contain the real value only 43%–61% of the time on held-out samples. Widening them to near-nominal coverage made the Bayesian loop find fewer top-5% samples (clearly fewer on 2 of 4 campaigns, see 'Is the model's uncertainty honest?'), so the optimizer keeps the overconfident default; only the ± shown to people is widened to the held-out calibration.
- **Claude's agent results are pending an API key** (see section 3).

