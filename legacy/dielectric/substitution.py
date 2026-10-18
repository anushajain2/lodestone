"""Generate new candidates by swapping in chemically similar elements (C4, stretch).

Starting from a promising measured material, substitute one element with chemically similar ones
(pymatgen's data-mined substitution probabilities, Hautier et al. 2011), keep compositions that are
charge-balanced and not already in the catalog, and score them with the program's surrogate on
composition features.

These are hypotheses: no DFT, no structure relaxation, no lab ground truth. The dashboard labels
them "generated, unverified" and they cannot be sent to the simulated labs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from pymatgen.core.periodic_table import Species


def _species(comp: Composition) -> list[Species] | None:
    guesses = comp.oxi_state_guesses(max_sites=-50)
    if not guesses:
        return None
    return [Species(el, ox) for el, ox in guesses[0].items()]


def propose_substitutions(formula: str, excluded: list[str], threshold: float = 1e-3, top: int = 15) -> list[dict]:
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor

    warnings.filterwarnings("ignore")
    comp = Composition(formula)
    species = _species(comp)
    if not species:
        return []
    out: dict[str, dict] = {}
    pred = SubstitutionPredictor(threshold=threshold)
    for sp in species:
        try:
            subs = pred.list_prediction([sp])
        except Exception:
            continue
        for s in subs:
            ((new, orig),) = s["substitutions"].items()  # pymatgen maps {substitute: original}
            if new.symbol in excluded or new.symbol == orig.symbol or new.oxi_state != orig.oxi_state:
                continue
            amounts: dict[str, float] = {}
            for el, amt in comp.items():
                key = new.symbol if el.symbol == orig.symbol else el.symbol
                amounts[key] = amounts.get(key, 0.0) + amt
            new_comp = Composition(amounts)
            f = new_comp.reduced_formula
            if f == comp.reduced_formula:
                continue
            if not new_comp.oxi_state_guesses(max_sites=-50):
                continue
            prev = out.get(f)
            if prev is None or s["probability"] > prev["probability"]:
                out[f] = {"formula": f, "parent": comp.reduced_formula, "swap": f"{orig.symbol}→{new.symbol}", "probability": float(s["probability"])}
    ranked = sorted(out.values(), key=lambda d: -d["probability"])
    return ranked[:top]


def score_generated(cands: list[dict], catalog: pd.DataFrame, surrogate, pool_ids: list[str]) -> list[dict]:
    """Predict k and gap for generated compositions using the program's surrogate.

    Generated compositions have no structure, so structure features are borrowed from the parent
    and the priors are recomputed from composition. Treat predictions as rough.
    """
    from lodestone.data.features import CURATED, load_features
    from lodestone.data.priors import _magpie  # noqa: F401  (ensures matminer present)
    from matminer.featurizers.composition import ElementProperty

    known = set(catalog["formula"])
    cands = [c for c in cands if c["formula"] not in known]
    if not cands or surrogate is None:
        return cands
    feats = load_features().loc[pool_ids]
    ep = ElementProperty.from_preset("magpie")
    labels = ep.feature_labels()
    cols = [c for c in CURATED if c in feats.columns]
    mu, sd = feats[cols].mean(), feats[cols].std() + 1e-9
    parent_rows = catalog.reset_index(drop=True).drop_duplicates("formula").set_index("formula")
    X = []
    for c in cands:
        v = dict(zip(labels, ep.featurize(Composition(c["formula"]))))
        parent_id = parent_rows.loc[c["parent"], "material_id"] if c["parent"] in parent_rows.index else None
        row = []
        for col in cols:
            if col in v:
                row.append(v[col])
            elif parent_id is not None:
                row.append(feats.loc[parent_id, col] if parent_id in feats.index else mu[col])
            else:
                row.append(mu[col])
        X.append((np.array(row) - mu.values) / sd.values)
    X = np.array(X)
    if surrogate.X_pool.shape[1] > X.shape[1]:
        # curated+priors: append prior predictions for the new compositions
        from lodestone.data.priors import load_priors

        pri = load_priors().loc[pool_ids]
        pg = (pri["prior_gap"].mean(), pri["prior_gap"].std())
        pe = (np.log(pri["prior_eps_inf"]).mean(), np.log(pri["prior_eps_inf"]).std())
        # No retraining here: use the parent's priors as a stand-in, flagged in the note.
        extra = []
        for c in cands:
            parent_id = parent_rows.loc[c["parent"], "material_id"] if c["parent"] in parent_rows.index else None
            g = pri.loc[parent_id, "prior_gap"] if parent_id in pri.index else pg[0]
            e = np.log(pri.loc[parent_id, "prior_eps_inf"]) if parent_id in pri.index else pe[0]
            extra.append([(g - pg[0]) / pg[1], (e - pe[0]) / pe[1]])
        X = np.hstack([X, np.array(extra)])
    m, s = surrogate.predict(X)
    for c, mm, ss in zip(cands, m, s):
        c.update({"k_pred": float(10 ** mm[0]), "gap_pred_eV": float(mm[1]), "gap_sd_eV": float(ss[1]), "status": "generated, unverified"})
    return sorted(cands, key=lambda d: -(d.get("k_pred", 0) * max(d.get("gap_pred_eV", 0), 0)))
