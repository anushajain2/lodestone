"""Experiment router (N2): send each experiment to a capable lab with the shortest queue."""

from __future__ import annotations

from dataclasses import dataclass

from lodestone.labs.mock_labs import LABS, capable_labs
from lodestone.schemas import LabSpec


class NoCapableLab(RuntimeError):
    pass


@dataclass
class RouteDecision:
    lab: LabSpec
    queue_before: int
    wait_days: float
    reason: str


def route(anion: str, queue_lengths: dict[str, int], measurements=("dielectric_constant", "band_gap")) -> RouteDecision:
    """Pick the capable lab with the shortest queue (expected wait = queue / capacity).

    Ties go to the faster turnaround, then the cheaper lab.

    ``queue_lengths`` is the live, shared view of every lab's outstanding jobs (counts only).
    """
    options = capable_labs(anion, measurements)
    if not options:
        raise NoCapableLab(f"no lab can make and measure a {anion}-anion compound")

    def wait(lab: LabSpec) -> float:
        return queue_lengths.get(lab.lab_id, 0) / lab.capacity_per_day

    best = min(options, key=lambda l: (round(wait(l), 2), l.turnaround_days, l.cost_per_experiment))
    others = ", ".join(f"{l.name.split()[0]} wait {wait(l):.1f}d" for l in options if l is not best)
    reason = f"{best.name}: capable ({anion}-compounds), shortest queue ({queue_lengths.get(best.lab_id, 0)} jobs, ~{wait(best):.1f} d wait)"
    if others:
        reason += f"; vs {others}"
    elif len(LABS) > 1:
        reason += "; only capable lab"
    return RouteDecision(lab=best, queue_before=queue_lengths.get(best.lab_id, 0), wait_days=wait(best), reason=reason)
