"""A campaign as an optimization problem over real samples (L1, L5).

The pool is every accepted record of one material system in the partner's workspace. The optimizer
sees only the *features* of each sample: the recipe, meaning its cation composition, deposition
temperature and pressure, and film thickness. The measured properties are the hidden truth that only
the lab replay reveals, one experiment at a time.

Scores turn a campaign goal into one number per sample so we can say what "the top 5%" means:

* maximize:     the objective (log10 for conductivity)
* constrained:  the objective if every constraint holds, otherwise infeasible
* window:       minus the distance to the target window (0 inside it)
* pareto:       no single score; hypervolume is used instead
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lodestone.schemas import PROPERTIES, CampaignSpec, ExperimentRecord

INFEASIBLE = -np.inf


def fmt_value(v: float) -> str:
    """Readable numbers: 8,870 / 2.45 / 0.0031."""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.3g}"
    return f"{v:.2g}"


def transform(prop: str, values: np.ndarray) -> np.ndarray:
    """Model space: log10 for properties spanning decades (conductivity)."""
    v = np.asarray(values, dtype=float)
    if PROPERTIES[prop]["log"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(v)
    return v


def untransform(prop: str, values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return 10**v if PROPERTIES[prop]["log"] else v


def short_label(rec: dict) -> str:
    comp = " ".join(f"{el}{f:.2f}" for el, f in rec["composition"].items())
    t = rec["synthesis"].get("deposition_temp_c")
    temp = f" · {t:.0f} °C" if t is not None else ""
    return f"{comp}{temp}"


@dataclass
class Problem:
    spec: CampaignSpec
    ids: list[str]
    labels: list[str]
    library: np.ndarray
    X: np.ndarray
    feature_names: list[str]
    truth: dict[str, np.ndarray]  # property → measured value (NaN when missing), natural units
    missing_reason: dict[str, list[str | None]]
    records: list[dict] = field(repr=False, default_factory=list)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_records(cls, records: list[dict] | list[ExperimentRecord], spec: CampaignSpec) -> "Problem":
        recs = [r.model_dump() if isinstance(r, ExperimentRecord) else r for r in records]
        recs = [r for r in recs if r["system"] == spec.system]
        if len(recs) < spec.batch_size + 5:
            raise ValueError(f"only {len(recs)} records for {spec.system} in this workspace; import more data first")
        metals = sorted({el for r in recs for el in r["composition"]})
        cols: dict[str, np.ndarray] = {f"x_{el}": np.array([r["composition"].get(el, 0.0) for r in recs]) for el in metals}
        temp = np.array([np.nan if r["synthesis"].get("deposition_temp_c") is None else r["synthesis"]["deposition_temp_c"] for r in recs], float)
        press = np.array([np.nan if r["synthesis"].get("growth_pressure_mtorr") is None else r["synthesis"]["growth_pressure_mtorr"] for r in recs], float)
        thick = np.array([r["measurements"].get("thickness", {}).get("value", np.nan) for r in recs], float)
        for name, arr in (("deposition_temp_c", temp), ("growth_pressure_mtorr", press), ("thickness_um", thick)):
            if np.isfinite(arr).any():
                arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
                cols[name] = arr
        names = [n for n, a in cols.items() if np.nanstd(a) > 1e-9]
        if len([n for n in names if n.startswith("x_")]) == len(metals) and len(metals) > 1:
            names.remove(f"x_{metals[-1]}")  # fractions sum to 1: the last one is redundant
        X = np.column_stack([cols[n] for n in names]) if names else np.zeros((len(recs), 1))
        truth, why = {}, {}
        for p in PROPERTIES:
            truth[p] = np.array([r["measurements"].get(p, {}).get("value", np.nan) for r in recs], float)
            why[p] = [r["missing"].get(p) for r in recs]
        return cls(
            spec=spec,
            ids=[r["record_id"] for r in recs],
            labels=[short_label(r) for r in recs],
            library=np.array([r["provenance"]["source_library_id"] for r in recs]),
            X=X,
            feature_names=names,
            truth=truth,
            missing_reason=why,
            records=recs,
        )

    @property
    def n(self) -> int:
        return len(self.ids)

    def libraries(self) -> dict[str, np.ndarray]:
        """Library id → indices of its samples, in a stable order. One library is one deposition run."""
        return {str(lib): np.flatnonzero(self.library == lib) for lib in sorted(set(self.library.tolist()), key=str)}

    # ------------------------------------------------------------------ scoring

    def objective_props(self) -> list[str]:
        s = self.spec
        return [s.window.property] if s.window else [o.property for o in s.objectives]

    def score_from_values(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Scalar score (higher is better) from property values in natural units; -inf if infeasible or unmeasured."""
        s = self.spec
        if s.kind == "pareto":
            raise ValueError("pareto campaigns use hypervolume, not a scalar score")
        if s.window:
            y = values[s.window.property]
            dist = np.maximum(s.window.low - y, 0) + np.maximum(y - s.window.high, 0)
            out = -dist
        else:
            o = s.objectives[0]
            out = transform(o.property, values[o.property]) * (1 if o.direction == "max" else -1)
        out = np.where(np.isfinite(out), out, INFEASIBLE)
        for c in s.constraints:
            v = values[c.property]
            ok = np.isfinite(v) & ((v >= c.value) if c.op == ">=" else (v <= c.value))
            out = np.where(ok, out, INFEASIBLE)
        return out

    def true_scores(self) -> np.ndarray:
        return self.score_from_values(self.truth)

    def top_set(self, fraction: float = 0.05) -> np.ndarray:
        """Indices of the best `fraction` of samples by true score (ties at the threshold included)."""
        sc = self.true_scores()
        finite = np.isfinite(sc)
        if not finite.any():
            return np.array([], dtype=int)
        k = max(1, int(round(fraction * self.n)))
        thr = np.sort(sc[finite])[::-1][min(k, finite.sum()) - 1]
        return np.flatnonzero(finite & (sc >= thr))

    def objective_matrix(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Pareto campaigns: objectives in model space with 'max' orientation."""
        return np.column_stack([
            transform(o.property, values[o.property]) * (1 if o.direction == "max" else -1) for o in self.spec.objectives
        ])

    def natural_best(self, score: float) -> str:
        """Human text for a score value."""
        s = self.spec
        if not np.isfinite(score):
            return "none feasible yet"
        if s.window:
            return "inside the window" if score >= 0 else f"{-score:.2f} {PROPERTIES[s.window.property]['unit']} outside the window"
        o = s.objectives[0]
        v = untransform(o.property, np.array([score * (1 if o.direction == "max" else -1)]))[0]
        return f"{fmt_value(v)} {PROPERTIES[o.property]['unit']}"
