"""Scores for the science loop: hypervolume, Pareto front, figure of merit."""

from __future__ import annotations

import numpy as np

# Hypervolume reference point in objective space (log10 k, band gap eV): roughly "better than SiO2's
# k and still a usable insulator". Only the region beyond it counts.
DEFAULT_REF = np.array([np.log10(3.9), 2.0])


def figure_of_merit(k, gap):
    """Simple single score tracked alongside the front: k × Eg (eV). Higher is better."""
    return np.asarray(k, float) * np.asarray(gap, float)


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """True for points no other point beats in both objectives (maximization)."""
    Y = np.asarray(Y, float)
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def hypervolume_2d(Y: np.ndarray, ref: np.ndarray = DEFAULT_REF) -> float:
    """Area dominated by the points and bounded by ref (both objectives maximized)."""
    Y = np.asarray(Y, float)
    if len(Y) == 0:
        return 0.0
    Y = Y[np.all(Y > ref, axis=1)]
    if len(Y) == 0:
        return 0.0
    Y = Y[pareto_mask(Y)]
    Y = Y[np.argsort(-Y[:, 0])]  # descending in objective 0 → ascending in objective 1 on the front
    hv, prev_y1 = 0.0, ref[1]
    for y0, y1 in Y:
        if y1 > prev_y1:
            hv += (y0 - ref[0]) * (y1 - prev_y1)
            prev_y1 = y1
    return float(hv)
