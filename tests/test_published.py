import math

import numpy as np
import pandas as pd
import pytest

from lodestone.connectors.adapters import LabCSVAdapter
from lodestone.connectors.mapping import ColumnMapping
from lodestone.loop import published


def test_p3ht_import_matches_the_papers_preprocessing():
    recs, imp = published.load()
    assert imp == {"received": 233, "accepted": 178, "rejected": 0, "system": "blend:D1-D2-D6-D8-P3HT"}
    raw = pd.read_csv(published.DATA / "P3HT_dataset.csv")
    feats, obj = list(raw.columns[:-1]), raw.columns[-1]
    paper = raw.groupby(feats)[obj].agg(lambda x: x.unique().mean())  # as in the paper's notebook
    n_top = math.ceil(0.05 * len(paper))
    ours = sorted(r["measurements"]["conductivity"]["value"] for r in recs)[-n_top:]
    assert ours == pytest.approx(sorted(paper.nlargest(n_top)))
    assert all(abs(sum(r["composition"].values()) - 1) < 1e-9 for r in recs)


def test_blend_mapping_rules():
    cols = {"composition": {"P3HT": "P", "D1": "D"}, "conductivity": "S [S/cm]"}
    with pytest.raises(ValueError, match="element symbols"):
        ColumnMapping(columns={"sample_id": "id", **cols})  # ingredient names need composition_kind: components
    with pytest.raises(ValueError, match="doesn't match"):
        ColumnMapping(columns=cols, composition_kind="components", row_ids=True, system="blend:D1-D2")
    m = ColumnMapping(columns=cols, composition_kind="components", row_ids=True, replicates="average")
    assert m.system == "blend:D1-P3HT"


def test_replicates_are_averaged_and_counted():
    m = ColumnMapping(columns={"composition": {"P3HT": "P", "D1": "D"}, "conductivity": "S [S/cm]"},
                      composition_kind="components", row_ids=True, replicates="average")
    text = "P,D,S [S/cm]\n50,50,10\n50.0,50,20\n80,20,5\n"
    recs, reps = LabCSVAdapter(m).adapt_csv(text)
    assert (reps[0].received, reps[0].accepted) == (3, 2)
    assert sorted(r.measurements["conductivity"].value for r in recs) == [5, 15]
    assert any(i.field == "replicates" and "row-1, row-2" in i.reason for i in reps[0].issues)


def test_random_baseline_is_linear_in_draws():
    r = published.random_top_pct(178, 9, 178)
    assert np.allclose(r, np.arange(1, 179) / 178)
