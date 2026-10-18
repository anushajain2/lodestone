"""The materials catalog: public computed data, pulled once and cached locally (C1).

Two sources produce the same columns:

* ``mp-api``    – live Materials Project pull (needs ``MP_API_KEY``).
* ``matminer``  – the published MP-derived snapshots bundled with matminer: the DFPT dielectric set
                  (Petousis et al., Sci. Data 2017, 1,056 compounds) joined to the 2018 MP release for
                  energy above hull. No key needed; this is what ships in ``data/catalog.json.gz``.

In the simulation, ``k_total`` and ``band_gap`` are the hidden "true" values the mock labs measure
(with noise). Nothing outside :mod:`lodestone.labs` reads them.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

from lodestone import config

CATALOG_COLUMNS = [
    "material_id",
    "formula",
    "elements",
    "n_elements",
    "anion",
    "nsites",
    "space_group",
    "e_hull",
    "k_total",
    "k_electronic",
    "k_ionic",
    "band_gap",
    "pot_ferroelectric",
    "structure",
    "source",
]

# Columns only the simulated labs (and the offline benchmark's scorer) may read.
HIDDEN_TRUTH_COLUMNS = ["k_total", "k_electronic", "k_ionic", "band_gap"]


def _anion(comp: Composition) -> str:
    """The most electronegative element: a crude but useful chemistry class (oxide, nitride, ...)."""
    els = [e for e in comp.elements if e.X == e.X]  # drop NaN electronegativity (noble gases)
    return max(els, key=lambda e: e.X).symbol if els else comp.elements[-1].symbol


def _finish(df: pd.DataFrame, source: str) -> pd.DataFrame:
    comps = [Composition(f) for f in df["formula"]]
    df["formula"] = [c.reduced_formula for c in comps]
    df["elements"] = [sorted(e.symbol for e in c.elements) for c in comps]
    df["n_elements"] = [len(c.elements) for c in comps]
    df["anion"] = [_anion(c) for c in comps]
    df["k_ionic"] = (df["k_total"] - df["k_electronic"]).clip(lower=0.0)
    df["source"] = source
    df = df.dropna(subset=["k_total", "band_gap"])
    df = df[(df["k_total"] >= 1.0)]  # a handful of DFPT entries are unphysical
    return df[CATALOG_COLUMNS].sort_values("material_id").reset_index(drop=True)


def build_from_matminer(data_home: str | os.PathLike = config.RAW_DIR) -> pd.DataFrame:
    from matminer.datasets import load_dataset

    diel = load_dataset("dielectric_constant", data_home=str(data_home))
    mp = load_dataset("mp_all_20181018", data_home=str(data_home))[["mpid", "e_hull"]]
    df = diel.merge(mp, left_on="material_id", right_on="mpid", how="left")
    df = df.rename(columns={"poly_total": "k_total", "poly_electronic": "k_electronic"})
    df["structure"] = [json.dumps(s.as_dict()) for s in df["structure"]]
    return _finish(df, "Materials Project via matminer (Petousis 2017 DFPT set + MP 2018 hull)")


def build_from_mp(api_key: str | None = None) -> pd.DataFrame:
    """Live pull from Materials Project. Needs a (free) API key in MP_API_KEY."""
    from mp_api.client import MPRester

    key = api_key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise RuntimeError("Set MP_API_KEY to pull from Materials Project (free key at materialsproject.org).")
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            has_props=["dielectric"],
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull",
                "e_total",
                "e_electronic",
                "nsites",
                "symmetry",
                "structure",
            ],
        )
    rows = []
    for d in docs:
        rows.append(
            {
                "material_id": str(d.material_id),
                "formula": d.formula_pretty,
                "band_gap": d.band_gap,
                "e_hull": d.energy_above_hull,
                "k_total": d.e_total,
                "k_electronic": d.e_electronic,
                "nsites": d.nsites,
                "space_group": d.symmetry.symbol if d.symmetry else None,
                "pot_ferroelectric": False,
                "structure": json.dumps(d.structure.as_dict()),
            }
        )
    return _finish(pd.DataFrame(rows), "Materials Project (live mp-api pull)")


def save_catalog(df: pd.DataFrame, path=config.CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", compression="gzip")
    load_catalog.cache_clear()


@lru_cache(maxsize=1)
def load_catalog(path=None) -> pd.DataFrame:
    path = path or config.CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run lodestone data build` (or `data pull-mp` with an MP key)."
        )
    df = pd.read_json(path, orient="records", compression="gzip", dtype={"material_id": str})
    df["e_hull"] = df["e_hull"].astype(float)
    return df.set_index("material_id", drop=False)


def material_exists(material_id: str) -> bool:
    return material_id in load_catalog().index


def structure_of(material_id: str) -> Structure:
    return Structure.from_dict(json.loads(load_catalog().loc[material_id, "structure"]))


def public_view(df: pd.DataFrame) -> pd.DataFrame:
    """The catalog without the hidden ground-truth columns."""
    return df.drop(columns=[c for c in HIDDEN_TRUTH_COLUMNS if c in df.columns])


def stats(df: pd.DataFrame | None = None) -> dict:
    """Answers the day-1 question: how many materials have every property we need?"""
    df = load_catalog() if df is None else df
    stable = df["e_hull"] <= config.DEFAULT_E_HULL_MAX
    return {
        "source": df["source"].iloc[0],
        "with_k_and_gap": int(len(df)),
        "with_k_gap_and_hull": int(df["e_hull"].notna().sum()),
        "stable_le_0.05": int(stable.sum()),
        "gap_gt_4": int((df["band_gap"] > 4).sum()),
        "k_gt_15": int((df["k_total"] > 15).sum()),
        "k_gt_15_and_gap_gt_4": int(((df["k_total"] > 15) & (df["band_gap"] > 4)).sum()),
        "k_gt_15_gap_gt_4_stable": int(((df["k_total"] > 15) & (df["band_gap"] > 4) & stable).sum()),
        "k_gt_20_and_gap_gt_5": int(((df["k_total"] > 20) & (df["band_gap"] > 5)).sum()),
        "anions": df["anion"].value_counts().head(8).to_dict(),
        "k_median": float(np.median(df["k_total"])),
        "gap_median": float(np.median(df["band_gap"])),
    }
