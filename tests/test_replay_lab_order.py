import csv
import io
from datetime import date, timedelta

import numpy as np

from lodestone.connectors.mapping import HELIX_MAPPING
from lodestone.replay_report import render_html, run_replay_report

HELIX = HELIX_MAPPING.parent / "helix_second_lab_export.csv"
GOAL = "maximize conductivity with band gap >= 3 eV"


def _rewrite_dates(tmp_path, how: str):
    rows = list(csv.DictReader(io.StringIO(HELIX.read_text(encoding="utf-8-sig"))))
    order = np.random.default_rng(0).permutation(len(rows))
    for rank, i in enumerate(order):
        rows[i]["Date"] = (date(2011, 1, 1) + timedelta(days=int(rank))).strftime("%d/%m/%Y") if how == "row" else ""
    out = tmp_path / f"helix_{how}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def test_row_dates_make_the_labs_order_the_headline(tmp_path):
    r = run_replay_report(_rewrite_dates(tmp_path, "row"), GOAL, HELIX_MAPPING, seeds=2, budget=20)
    lab = r["lab"]
    assert lab["level"] == "row" and lab["usable"] and r["headline"] == "lab"
    assert r["verdict"] == lab["verdict"] and lab["unit"] == "experiments"
    assert set(lab["methods"]) == {"lab-order", "bayes-gp-from-lab-start"}
    page = render_html(r)
    assert "saved vs what your lab actually did" in page and "Your lab&#x27;s actual order" in page


def test_run_dates_with_few_runs_keep_random_as_the_headline():
    r = run_replay_report(HELIX, GOAL, HELIX_MAPPING, seeds=2, budget=20)
    assert r["lab"]["level"] == "run" and not r["lab"]["usable"] and r["lab"]["verdict"] == "too_few"
    assert r["headline"] == "random"
    assert "too few to tell" in render_html(r)


def test_no_dates_says_the_baselines_flatter_the_model(tmp_path):
    r = run_replay_report(_rewrite_dates(tmp_path, "none"), GOAL, HELIX_MAPPING, seeds=2, budget=20)
    assert r["lab"]["level"] == "none" and r["headline"] == "random"
    assert "no usable dates" in render_html(r)
