"""Gaussian-process surrogate on composition features (L1).

Two outputs, modelled jointly by a batched GP: y0 = log10(k) and y1 = band gap (eV). We model log k
because DFPT dielectric constants span two orders of magnitude and are right-skewed.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)


def to_objectives(k: np.ndarray, gap: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log10(np.asarray(k, float)), np.asarray(gap, float)])


def from_objectives(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 10 ** y[:, 0], y[:, 1]


class Surrogate:
    """Fit on tested materials; predict for the rest of the pool."""

    def __init__(self, X_pool: np.ndarray):
        self.X_pool = torch.as_tensor(X_pool)
        lo, hi = self.X_pool.min(0).values, self.X_pool.max(0).values
        self.bounds = torch.stack([lo, torch.where(hi > lo, hi, lo + 1.0)])
        self.model: SingleTaskGP | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Surrogate":
        X_t, Y_t = torch.as_tensor(X), torch.as_tensor(Y)
        model = SingleTaskGP(
            X_t,
            Y_t,
            input_transform=Normalize(d=X_t.shape[-1], bounds=self.bounds),
            outcome_transform=Standardize(m=Y_t.shape[-1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(mll)
            except Exception:  # keep the loop alive on rare optimizer failures; priors still give a usable model
                pass
        model.eval()
        self.model = model
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation (latent function, objective space)."""
        assert self.model is not None, "fit() first"
        post = self.model.posterior(torch.as_tensor(X))
        mean = post.mean.numpy()
        std = post.variance.clamp_min(1e-12).sqrt().numpy()
        return mean, std

    def lengthscales(self) -> np.ndarray | None:
        try:
            return self.model.covar_module.lengthscale.detach().numpy()  # type: ignore[union-attr]
        except Exception:
            return None
