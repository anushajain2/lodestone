"""Lab replay (L5): run an experiment by returning that sample's real HTEM measurements.

Two kinds of failure, both recorded:

* **real**: the measurement doesn't exist in the data (e.g. no four-point-probe reading because the
  film was too resistive, no Tauc fit). Replaying a real gap gives a real failure, with the reason.
* **simulated**: a configurable rate of lab mishaps (delaminated film, stage error, lost sample) on
  top, to mimic a working lab. A simulated failure can be retried once.

Delays are in rounds: a batch submitted in round r comes back in round r + delay.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from lodestone.loop.problem import Problem

MISHAPS = [
    "film delaminated during handling (simulated)",
    "XRF stage error, sample not measured (simulated)",
    "sample lost in transfer between instruments (simulated)",
    "four-point-probe contact failure (simulated)",
]


def seeded_rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


@dataclass
class Outcome:
    status: str  # ok | failed
    values: dict[str, float | None]
    kind: str | None = None  # missing | random
    reason: str | None = None


class LabReplay:
    def __init__(self, problem: Problem, failure_rate: float, seed: int):
        self.problem = problem
        self.failure_rate = failure_rate
        self.seed = seed

    def run(self, i: int, attempt: int = 1) -> Outcome:
        rng = seeded_rng(self.seed, self.problem.ids[i], attempt)
        if rng.random() < self.failure_rate:
            return Outcome("failed", {}, "random", str(rng.choice(MISHAPS)))
        needed = self.problem.spec.properties_needed()
        values = {p: (None if not np.isfinite(self.problem.truth[p][i]) else float(self.problem.truth[p][i])) for p in self.problem.truth}
        miss = [p for p in needed if values[p] is None]
        if miss:
            reasons = "; ".join(f"{p}: {self.problem.missing_reason[p][i] or 'not measured'}" for p in miss)
            return Outcome("failed", values, "missing", reasons)
        return Outcome("ok", values)
