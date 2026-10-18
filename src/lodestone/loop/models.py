"""Surrogate models (L2) and how failed experiments feed them (L6).

* ``gp``: one BoTorch ``SingleTaskGP`` per property (each property has its own training set,
  because a sample can have a band gap but no conductivity), combined in a ``ModelListGP``.
* ``rf``: scikit-learn extra-trees per property; the spread across trees is the uncertainty.

Failed experiments are not dropped:

* A missing four-point-probe reading in HTEM almost always means the film was too resistive for the
  instrument. That is information: it is fed to the conductivity model as a censored low value,
  one decade below the lowest conductivity measured so far in the campaign.
* A logistic "will this measurement succeed?" model is trained on every completed experiment
  (random lab mishaps excluded, since they carry no information about the recipe) and multiplies the
  acquisition value, so the loop learns to avoid regions where the measurement keeps failing.

GP options (``GP_OPTIONS``, or the ``gp_options`` context manager for experiments):

* ``noise_floor``: lower bound on the noise variance, in standardized units.
* ``kernel``: ``"rbf"`` (BoTorch's default) or ``"matern52"`` with an outputscale and a lengthscale prior.
* ``calibrate``: inflate each property's posterior std by a factor from K-fold cross-validation with
  the hyperparameters refit in every fold. Scaling the kernel and the noise by the same c² leaves the
  mean unchanged and multiplies the posterior variance by c². See ``loop/calibration.py`` for the
  held-out check and why these are off by default.
* ``std_inflation``: a fixed factor per property ({"conductivity": 1.6, ...}) instead of the CV one.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression

from lodestone.loop.problem import transform

torch.set_default_dtype(torch.float64)

CENSOR_DECADES = 1.0
DEFAULT_LOG_FLOOR = -3.0

GP_OPTIONS: dict = {"noise_floor": None, "kernel": "rbf", "calibrate": False, "cv_folds": 5, "std_inflation": None}
MIN_POINTS_TO_CALIBRATE = 10
MAX_INFLATION = 10.0


@contextmanager
def gp_options(**kw):
    """Temporarily change the GP defaults (for experiments and the calibration check)."""
    old = dict(GP_OPTIONS)
    GP_OPTIONS.update(kw)
    try:
        yield
    finally:
        GP_OPTIONS.clear()
        GP_OPTIONS.update(old)


@dataclass
class Observations:
    """What the optimizer has learned so far (natural units)."""

    order: list[int] = field(default_factory=list)  # completed experiments, in arrival order
    values: dict[str, dict[int, float]] = field(default_factory=dict)
    failures: list[tuple[int, str, str]] = field(default_factory=list)  # (index, kind, reason); kind: random|missing

    def add(self, i: int, values: dict[str, float | None], failed_kind: str | None = None, reason: str | None = None) -> None:
        self.order.append(i)
        for p, v in values.items():
            if v is not None and np.isfinite(v):
                self.values.setdefault(p, {})[i] = float(v)
        if failed_kind:
            self.failures.append((i, failed_kind, reason or ""))

    def tested(self) -> set[int]:
        return set(self.order)

    def missing_failures(self, prop: str | None = None) -> list[int]:
        return [i for i, kind, reason in self.failures if kind == "missing" and (prop is None or prop in reason)]


def training_data(obs: Observations, props: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Per property: (indices, model-space targets, number of censored points)."""
    out = {}
    for p in props:
        vals = obs.values.get(p, {})
        idx = np.array(list(vals), dtype=int)
        y = transform(p, np.array(list(vals.values()), dtype=float)) if len(idx) else np.zeros(0)
        n_cens = 0
        if p == "conductivity":
            cens = [i for i in obs.missing_failures("conductivity") if i not in vals]
            if cens:
                floor = (y.min() - CENSOR_DECADES) if len(y) else DEFAULT_LOG_FLOOR
                idx = np.concatenate([idx, np.array(cens, dtype=int)])
                y = np.concatenate([y, np.full(len(cens), floor)])
                n_cens = len(cens)
        out[p] = (idx, y, n_cens)
    return out


class Surrogate:
    def __init__(self, X_pool: np.ndarray, kind: str = "gp", seed: int = 0):
        self.X_pool = np.asarray(X_pool, dtype=float)
        lo, hi = self.X_pool.min(0), self.X_pool.max(0)
        self.bounds = torch.as_tensor(np.stack([lo, np.where(hi > lo, hi, lo + 1.0)]))
        self.kind = kind
        self.seed = seed
        self.props: list[str] = []
        self.model = None  # ModelListGP for gp
        self.forests: dict[str, ExtraTreesRegressor] = {}
        self.options = dict(GP_OPTIONS)
        self.std_scale: dict[str, float] = {}  # per property, 1.0 unless calibrate is on

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> SingleTaskGP:
        Xt, yt = torch.as_tensor(X), torch.as_tensor(y).unsqueeze(-1)
        kw = {}
        if self.options["noise_floor"] is not None:
            kw["likelihood"] = GaussianLikelihood(noise_constraint=GreaterThan(self.options["noise_floor"]))
        if self.options["kernel"] == "matern52":
            d = Xt.shape[-1]
            kw["covar_module"] = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=LogNormalPrior(1.4 + np.log(d) / 2, 1.73**0.5)),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        elif self.options["kernel"] != "rbf":
            raise ValueError(f"unknown kernel {self.options['kernel']!r}")
        m = SingleTaskGP(Xt, yt, input_transform=Normalize(d=Xt.shape[-1], bounds=self.bounds), outcome_transform=Standardize(m=1), **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
            except Exception:  # keep the campaign alive on a rare optimizer failure; the priors still give a usable model
                pass
        return m.eval()

    @torch.no_grad()
    def cv_std_scale(self, X: np.ndarray, y: np.ndarray) -> float:
        """Factor c such that 68% of K-fold held-out values lie within ±c·σ (σ includes the fitted noise)."""
        n = len(X)
        folds = np.array_split(np.random.default_rng(n).permutation(n), min(self.options["cv_folds"], n))
        z = []
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            post = self._fit_gp(X[tr], y[tr]).posterior(torch.as_tensor(X[te]), observation_noise=True)
            z.append(np.abs(y[te] - post.mean.squeeze(-1).numpy()) / post.variance.squeeze(-1).clamp_min(1e-12).sqrt().numpy())
        return float(np.clip(np.quantile(np.concatenate(z), 0.6827), 1.0, MAX_INFLATION))

    def _fit_calibrated_gp(self, X: np.ndarray, y: np.ndarray, prop: str) -> SingleTaskGP:
        m = self._fit_gp(X, y)
        fixed = self.options["std_inflation"] or {}
        if prop in fixed:
            c = float(fixed[prop])
        elif self.options["calibrate"] and len(X) >= MIN_POINTS_TO_CALIBRATE:
            c = self.cv_std_scale(X, y)
        else:
            c = 1.0
        self.std_scale[prop] = c
        if c > 1.0:
            noise = m.likelihood.noise.detach().clone()
            kernel = ScaleKernel(m.covar_module)
            kernel.outputscale = torch.tensor(c**2)
            m.covar_module = kernel
            m.likelihood.noise = noise * c**2
            m.prediction_strategy = None
        return m

    def fit(self, data: dict[str, tuple[np.ndarray, np.ndarray, int]]) -> "Surrogate":
        self.props = list(data)
        if self.kind == "gp":
            torch.manual_seed(self.seed)
            models = []
            for p in self.props:
                idx, y, _ = data[p]
                if len(idx) < 2:
                    raise ValueError(f"need at least 2 observations of {p} to fit a model")
                models.append(self._fit_calibrated_gp(self.X_pool[idx], y, p))
            self.model = ModelListGP(*models).eval()
        elif self.kind == "rf":
            for p in self.props:
                idx, y, _ = data[p]
                if len(idx) < 2:
                    raise ValueError(f"need at least 2 observations of {p} to fit a model")
                f = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
                self.forests[p] = f.fit(self.X_pool[idx], y)
        else:
            raise ValueError(f"unknown surrogate {self.kind!r}")
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Mean and std per property, in model space."""
        X = np.asarray(X, dtype=float)
        out = {}
        if self.kind == "gp":
            post = self.model.posterior(torch.as_tensor(X))
            mean = post.mean.numpy()
            std = post.variance.clamp_min(1e-12).sqrt().numpy()
            for j, p in enumerate(self.props):
                out[p] = (mean[:, j], std[:, j])
        else:
            for p, f in self.forests.items():
                per_tree = np.stack([t.predict(X) for t in f.estimators_])
                out[p] = (per_tree.mean(0), np.maximum(per_tree.std(0), 1e-3))
        return out


class SuccessModel:
    """P(the needed measurements succeed | recipe), learned from real measurement failures."""

    def __init__(self, X_pool: np.ndarray):
        self.mu = X_pool.mean(0)
        self.sd = X_pool.std(0) + 1e-9
        self.clf: LogisticRegression | None = None

    def fit(self, obs: Observations, needed: list[str], X_pool: np.ndarray) -> "SuccessModel":
        random_fail = {i for i, kind, _ in obs.failures if kind == "random"}
        idx, lab = [], []
        for i in dict.fromkeys(obs.order):
            if i in random_fail and not any(i in obs.values.get(p, {}) for p in needed):
                continue
            idx.append(i)
            lab.append(int(all(i in obs.values.get(p, {}) for p in needed)))
        lab = np.array(lab)
        if len(lab) and lab.sum() >= 3 and (len(lab) - lab.sum()) >= 3:
            self.clf = LogisticRegression(C=0.5, max_iter=500).fit((X_pool[idx] - self.mu) / self.sd, lab)
        return self

    def p_success(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            return np.ones(len(X))
        return np.clip(self.clf.predict_proba((X - self.mu) / self.sd)[:, 1], 0.02, 1.0)
