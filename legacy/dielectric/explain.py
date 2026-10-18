"""'Why proposed' notes (D2): short, grounded reasons built from the model's own numbers."""

from __future__ import annotations

import numpy as np

from lodestone.loop.metrics import pareto_mask
from lodestone.schemas import TargetSpec

_ANION_NAMES = {"O": "oxide", "N": "nitride", "F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide", "S": "sulfide", "Se": "selenide", "Te": "telluride", "P": "phosphide", "H": "hydride"}


def chemistry_label(anion: str) -> str:
    return _ANION_NAMES.get(anion, f"{anion}-anion compound")


def why_proposed(
    *,
    formula: str,
    anion: str,
    e_hull: float | None,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    strategy: str,
    spec: TargetSpec,
    Y_tested: np.ndarray,
    pool_std_median: np.ndarray | None,
    neighbour: tuple[str, float, float] | None,
    extra_notes: list[str] | None = None,
    initial: bool = False,
) -> str:
    stab = "on the convex hull" if (e_hull is not None and e_hull <= 1e-6) else (
        f"E_hull {e_hull:.3f} eV/atom" if e_hull is not None else "stability unknown"
    )
    chem = chemistry_label(anion)
    if initial:
        return f"Initial space-filling pick: a {chem} far (in composition-feature space) from the other seed experiments, so the first model sees varied chemistry. {stab.capitalize()}."
    if strategy == "random" or mean is None:
        return f"Random baseline pick ({chem}, {stab})."

    k_mid, k_lo, k_hi = 10 ** mean[0], 10 ** (mean[0] - std[0]), 10 ** (mean[0] + std[0])
    gap, gap_sd = mean[1], std[1]
    parts = [f"Predicted k ≈ {k_mid:.0f} ({k_lo:.0f}–{k_hi:.0f}) and gap ≈ {gap:.1f} ± {gap_sd:.1f} eV"]

    reasons = []
    if len(Y_tested):
        front = Y_tested[pareto_mask(Y_tested)]
        beyond = not np.any(np.all(front >= mean, axis=1))
        if beyond:
            reasons.append("would push the trade-off front outward if the prediction holds")
    if pool_std_median is not None:
        rel = float(np.mean(std / np.maximum(pool_std_median, 1e-9)))
        if rel > 1.25:
            reasons.append("the model is unsure here, so testing it teaches the model the most")
        elif rel < 0.8:
            reasons.append("the model is fairly confident here")
    meets = spec.is_hit(k_mid, gap)
    if (spec.k_min is not None or spec.gap_min_eV is not None) and meets:
        reasons.append("the central prediction meets your thresholds")
    if strategy == "greedy":
        reasons = ["highest predicted k×Eg (greedy baseline, ignores uncertainty)"]
    elif not reasons:
        reasons.append("among the best expected improvements to the trade-off front that are left, balancing predicted value against uncertainty")
    if reasons:
        parts.append("; ".join(reasons))
    text = ". ".join(p[0].upper() + p[1:] for p in parts) + "."
    if neighbour:
        nf, nk, ng = neighbour
        text += f" Closest tested material: {nf} (measured k {nk:.0f}, gap {ng:.1f} eV)."
    text += f" {chem.capitalize()}, {stab}."
    for n in extra_notes or []:
        text += f" {n[0].upper() + n[1:]}."
    return text
