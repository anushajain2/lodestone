"""Transfer 'prior' features from larger public datasets.

The loop starts with almost no data, so the surrogate gets two extra inputs learned elsewhere:

* ``prior_gap``     – a composition model trained on Matbench ``mp_gap`` (106k MP PBE band gaps)
* ``prior_eps_inf`` – a composition model trained on Matbench ``dielectric`` (4.7k MP refractive
                      indices), predicting ε∞ = n², the electronic part of k

Every formula that appears in our catalog — every polymorph — is removed from the training data
first, so the priors never see the materials the loop is searching. The resulting two columns are
small and ship in ``data/priors.parquet``; rebuilding them needs ~140 MB of downloads.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from lodestone import config
from lodestone.data.catalog import load_catalog

PRIORS_PATH = config.DATA_DIR / "priors.parquet"


def _magpie(formulas: list[str]) -> np.ndarray:
    from matminer.featurizers.composition import ElementProperty

    fz = ElementProperty.from_preset("magpie")
    fz.set_n_jobs(1)
    rows = []
    for f in formulas:
        try:
            rows.append(fz.featurize(Composition(f)))
        except Exception:
            rows.append([np.nan] * len(fz.feature_labels()))
    return np.array(rows, dtype=float)


def _fit_predict(train_formulas, y, query_formulas, label: str, verbose=True):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    Xtr, Xq = _magpie(train_formulas), _magpie(query_formulas)
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63, random_state=0)
    if verbose:
        idx = np.random.default_rng(0).choice(len(Xtr), size=min(len(Xtr), 20000), replace=False)
        cv = cross_val_score(model, Xtr[idx], y[idx], cv=3, scoring="r2")
        print(f"{label}: trained on {len(Xtr)} formulas, CV R² {cv.mean():.2f}")
    model.fit(Xtr, y)
    return model.predict(Xq)


def build_priors(verbose: bool = True) -> pd.DataFrame:
    from matminer.datasets import load_dataset

    warnings.filterwarnings("ignore")
    cat = load_catalog()
    excluded = set(cat["formula"])

    gap = load_dataset("matbench_mp_gap", data_home=str(config.RAW_DIR))
    gap["formula"] = [s.composition.reduced_formula for s in gap["structure"]]
    gap = gap[~gap["formula"].isin(excluded)].groupby("formula", as_index=False)["gap pbe"].median()

    diel = load_dataset("matbench_dielectric", data_home=str(config.RAW_DIR))
    diel["formula"] = [s.composition.reduced_formula for s in diel["structure"]]
    diel = diel[~diel["formula"].isin(excluded)].groupby("formula", as_index=False)["n"].median()

    query = list(cat["formula"])
    out = pd.DataFrame(index=cat.index)
    out["prior_gap"] = np.clip(_fit_predict(list(gap["formula"]), gap["gap pbe"].values, query, "band-gap prior", verbose), 0, None)
    out["prior_eps_inf"] = np.clip(
        np.exp(_fit_predict(list(diel["formula"]), np.log(diel["n"].values ** 2), query, "ε∞ prior (log n²)", verbose)), 1, None
    )
    out.to_parquet(PRIORS_PATH)
    load_priors.cache_clear()
    return out


@lru_cache(maxsize=1)
def load_priors() -> pd.DataFrame | None:
    if not PRIORS_PATH.exists():
        return None
    return pd.read_parquet(PRIORS_PATH)
