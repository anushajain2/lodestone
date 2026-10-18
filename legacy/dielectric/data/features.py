"""Composition (and a few cheap structure) features from matminer, cached to disk (L1 input).

The surrogate works on a compact version of these: standardized, then reduced with PCA fitted on
the whole candidate pool. PCA is unsupervised and the pool's features are public, so this leaks
nothing about the hidden property values.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lodestone import config
from lodestone.data.catalog import load_catalog


def compute_features(catalog: pd.DataFrame) -> pd.DataFrame:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
    from matminer.featurizers.structure import DensityFeatures

    warnings.filterwarnings("ignore")
    comps = [Composition(f) for f in catalog["formula"]]
    blocks = []
    for fz in (ElementProperty.from_preset("magpie"), Stoichiometry(), ValenceOrbital()):
        fz.set_n_jobs(1)
        rows = [fz.featurize(c) for c in comps]
        blocks.append(pd.DataFrame(rows, columns=fz.feature_labels(), index=catalog.index))
    dens = DensityFeatures()
    structs = [Structure.from_dict(json.loads(s)) for s in catalog["structure"]]
    rows = []
    for s in structs:
        try:
            rows.append(dens.featurize(s))
        except Exception:
            rows.append([np.nan] * len(dens.feature_labels()))
    blocks.append(pd.DataFrame(rows, columns=dens.feature_labels(), index=catalog.index))
    feats = pd.concat(blocks, axis=1).astype(float)
    feats = feats.loc[:, feats.std() > 1e-9]
    return feats.fillna(feats.median())


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    cat = load_catalog()
    path = config.FEATURES_PATH
    if path.exists():
        feats = pd.read_parquet(path)
        if set(feats.index) >= set(cat.index):
            return feats.loc[cat.index]
    feats = compute_features(cat)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path)
    return feats


_M = "MagpieData "
# A compact, chemically motivated set: bonding (electronegativity), valence shells, size, and the
# elemental band gaps. Small enough for a GP with one lengthscale per input to learn from ~30 points.
CURATED = [
    _M + "range Electronegativity", _M + "mean Electronegativity", _M + "maximum Electronegativity",
    _M + "minimum Electronegativity", _M + "mean GSbandgap", _M + "maximum GSbandgap", _M + "mean NpValence",
    _M + "mean NdValence", _M + "mean NsValence", _M + "mean NfValence", _M + "mean CovalentRadius",
    _M + "range CovalentRadius", _M + "mean Row", _M + "maximum Row", _M + "mean MendeleevNumber",
    _M + "range MendeleevNumber", _M + "mean NUnfilled", "density", "vpa", "packing fraction",
]
FEATURE_SETS = ("curated+priors", "curated", "pca12")


def surrogate_features(material_ids: list[str] | pd.Index, feature_set: str = "curated+priors") -> pd.DataFrame:
    """Model inputs for a pool, standardized over that pool. Indexed by material_id.

    * ``curated+priors`` (default): the curated set plus the two transfer priors (see priors.py)
    * ``curated``:                  the curated set alone
    * ``pca12``:                    all 140 matminer features, standardized, PCA to 12 components
    """
    feats = load_features().loc[list(material_ids)]
    if feature_set == "pca12":
        x = StandardScaler().fit_transform(feats.values)
        n = min(12, x.shape[0] - 1, x.shape[1])
        z = PCA(n_components=n, random_state=0).fit_transform(x)
        return pd.DataFrame(z, index=feats.index, columns=[f"pc{i}" for i in range(n)])
    cols = [c for c in CURATED if c in feats.columns]
    x = feats[cols].copy()
    if feature_set == "curated+priors":
        from lodestone.data.priors import load_priors

        pri = load_priors()
        if pri is None:
            raise FileNotFoundError("data/priors.parquet missing; run `uv run lodestone data priors`")
        x["prior_gap"] = pri.loc[x.index, "prior_gap"]
        x["prior_log_eps_inf"] = np.log(pri.loc[x.index, "prior_eps_inf"])
    elif feature_set != "curated":
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
    z = StandardScaler().fit_transform(x.values)
    return pd.DataFrame(z, index=x.index, columns=list(x.columns))


def reduced_features(material_ids: list[str] | pd.Index, feature_set: str = "curated+priors") -> pd.DataFrame:
    """Backwards-compatible name for :func:`surrogate_features`."""
    return surrogate_features(material_ids, feature_set)


def nearest_neighbours(material_id: str, among: list[str], k: int = 1) -> list[tuple[str, float]]:
    """Closest materials in standardized feature space (used for 'why proposed' notes)."""
    if not among:
        return []
    feats = load_features()
    scaled = (feats - feats.mean()) / (feats.std() + 1e-9)
    a = scaled.loc[material_id].values
    others = scaled.loc[among].values
    d = np.linalg.norm(others - a, axis=1)
    order = np.argsort(d)[:k]
    return [(among[i], float(d[i])) for i in order]
