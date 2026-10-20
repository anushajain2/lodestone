"""Five more HTEM systems, chosen by rule (E1, out-of-sample check).

The four demo campaigns were built on three systems picked while the method was being developed, so
a reader can fairly ask whether those systems were the ones where it happened to work. This module
reruns the same comparison on systems chosen by fixed rules, written down before any of their data
was downloaded or any optimization was run on them:

* **Systems.** The five HTEM DB systems with the most public libraries that have both electrical
  (four-point probe) and optical measurements, excluding the three demo systems, and that have at
  least ``MIN_USABLE`` imported samples with a conductivity value. ``select_systems`` recomputes it.

  The usability condition was added after downloading, before any optimization was run: the first
  rule (largest five by library count) picked Ba-Cu-S and Ba-Cu-S-Sr, which turned out to have no
  usable data (see ``EXCLUDED``). Downloading them also exposed an importer bug: for sulfides, XRF
  reports the sulfur too, and the importer rejected those records because the metals alone sum to
  ~50%. That is fixed in ``connectors.adapters.cation_fractions``; the demo systems are unchanged.
* **Goals.** "Most conductive" for every system. A transparent-conductor goal (maximize conductivity
  with direct band gap >= 3 eV, as in the demo) only where at least ``MIN_FEASIBLE`` accepted samples
  have both measurements and a gap >= 3 eV, so that its top 5% holds at least five samples. This
  depends on data availability only, not on any optimization result.
* **Settings.** Exactly the demo's: 60-experiment budget, 10 random seed experiments, batches of 5,
  5% simulated lab failures, 20 seeds, the same methods. The per-deposition-run comparison uses the
  existing run-level design (10 runs, 2 seed runs) and its rule of at least 20 libraries.

Results: ``reports/benchmark/more_systems.json`` and ``more_systems_library.json``; the report
section 'Five more systems, chosen by rule'.
"""

from __future__ import annotations

import json

import numpy as np

from lodestone import config
from lodestone.connectors import htem
from lodestone.loop.problem import transform
from lodestone.schemas import CampaignSpec

DEMO_SYSTEMS = ("N-Sn-Zn", "Co-Ni-O-Zn", "In-O-Sn-Zn")
N_SYSTEMS = 5
MIN_USABLE = 100  # imported samples with a conductivity value
MIN_FEASIBLE = 100
SYSTEMS = ("Cu-S-Sn", "Cu-O-Zn", "O-Sn-Ti-Zn", "Cu-S-Sb", "Cu-O-Se-Zn")
# Ranked above the cut-off but unusable (counts from the 2026-10-20 download, after the sulfide fix).
EXCLUDED = {
    "Ba-Cu-S": "1,320 samples served, 220 imported, only 8 with a conductivity value (no film thickness for the rest)",
    "Ba-Cu-S-Sr": "748 samples served, none with an XRF composition",
    "Al-Cu-S": "no samples with an XRF composition (tied on library count with Cu-O-Se-Zn, ranked first by name)",
}

NAMES = {
    "Cu-S-Sn": "copper tin sulfide",
    "Cu-O-Zn": "copper zinc oxide",
    "O-Sn-Ti-Zn": "zinc tin titanium oxide",
    "Cu-S-Sb": "copper antimony sulfide",
    "Cu-O-Se-Zn": "copper zinc oxide selenide",
}


def ranked_systems() -> list[tuple[str, int]]:
    """HTEM systems by number of libraries with electrical + optical data (demo systems excluded)."""
    counts: dict[str, int] = {}
    for lib in htem.list_libraries():
        key = htem.system_key(lib.get("elements"))
        if key and key not in DEMO_SYSTEMS and lib.get("has_ele") and lib.get("has_opt"):
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def usable(system: str) -> int:
    """Imported samples with a conductivity value (needs the system's snapshot)."""
    from lodestone.loop.benchmark import public_records

    return sum(1 for r in public_records(system) if "conductivity" in r["measurements"])


def select_systems(n: int = N_SYSTEMS) -> list[str]:
    """The selection rule, recomputed (pulls nothing: every candidate up to the cut-off must have a snapshot)."""
    out = []
    for system, _ in ranked_systems():
        if usable(system) >= MIN_USABLE:
            out.append(system)
        if len(out) == n:
            break
    return out


def feasible_transparent(records: list[dict]) -> int:
    """Accepted samples with a measured conductivity and a direct band gap >= 3 eV."""
    n = 0
    for r in records:
        m = r["measurements"]
        if "conductivity" in m and "band_gap" in m and m["band_gap"]["value"] >= 3.0:
            n += 1
    return n


def campaigns(records_for=None) -> dict[str, CampaignSpec]:
    """The campaigns the goal rule gives for SYSTEMS (needs their snapshots in data/htem/)."""
    from lodestone.loop.benchmark import public_records

    records_for = records_for or public_records
    out = {}
    for system in SYSTEMS:
        slug = system.lower()
        name = NAMES.get(system, system)
        out[f"{slug}-conductivity"] = CampaignSpec(name=f"Most conductive {name}", system=system, objectives=[{"property": "conductivity"}])
        if feasible_transparent(records_for(system)) >= MIN_FEASIBLE:
            out[f"{slug}-transparent-conductor"] = CampaignSpec(
                name=f"Transparent conductor, {name} (gap ≥ 3 eV)", system=system,
                objectives=[{"property": "conductivity"}], constraints=[{"property": "band_gap", "op": ">=", "value": 3.0}],
            )
    return out


def data_summary() -> dict:
    """What the import accepted per system, and how the goal rule came out."""
    from lodestone.loop.benchmark import public_records

    rows = {}
    for system in SYSTEMS:
        recs = public_records(system)
        libs = htem.load_snapshot(system)
        cond = np.array([r["measurements"]["conductivity"]["value"] for r in recs if "conductivity" in r["measurements"]], float)
        gap = np.array([r["measurements"]["band_gap"]["value"] for r in recs if "band_gap" in r["measurements"]], float)
        rows[system] = {
            "libraries": len(libs),
            "samples_received": int(sum(len(lib.get("samples") or []) for lib in libs)),
            "accepted": len(recs),
            "with_conductivity": int(len(cond)),
            "with_band_gap": int(len(gap)),
            "conductivity_decades": float(np.ptp(transform("conductivity", cond))) if len(cond) else None,
            "transparent_feasible": feasible_transparent(recs),
            "transparent_goal": feasible_transparent(recs) >= MIN_FEASIBLE,
        }
    return rows


def run(seeds: int = 20, verbose: bool = True) -> dict:
    """Per-sample and per-deposition-run benchmarks for every rule-chosen campaign."""
    from lodestone.loop.benchmark import METHODS, problem_for, register_campaigns, run_library_benchmark, run_problem
    from lodestone.loop.engine import top_found

    camps = campaigns()
    register_campaigns(camps)
    out = {"rules": {"systems": f"the {N_SYSTEMS} HTEM systems with the most libraries with electrical + optical data, excluding the demo systems "
                                f"({', '.join(DEMO_SYSTEMS)}), that have at least {MIN_USABLE} imported samples with a conductivity value",
                     "goals": f"most conductive for all; transparent conductor (gap >= 3 eV) where >= {MIN_FEASIBLE} samples qualify"},
           "systems": list(SYSTEMS), "excluded": EXCLUDED, "data": data_summary(), "config": {"seeds": seeds, "failure_rate": 0.05}, "campaigns": {}}
    for key, spec in camps.items():
        prob = problem_for(key, 0.05)
        summary, _, trajs = run_problem(prob, seeds, METHODS, key, verbose)
        top = prob.top_set(0.05)
        found = {}
        for t in trajs:
            found.setdefault(t.method, {})[t.seed] = top_found(prob, t, top)
        base = found["random"]
        for m, by_seed in found.items():
            d = np.array([by_seed[s] - base[s] for s in sorted(by_seed)], float)
            summary["methods"][m]["top_found_minus_random_mean"] = float(d.mean())
            summary["methods"][m]["top_found_minus_random_sem"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
        summary["spec"] = spec.model_dump()
        summary["summary"] = spec.summary()
        out["campaigns"][key] = summary
    path = config.REPORTS_DIR / "benchmark"
    path.mkdir(parents=True, exist_ok=True)
    (path / "more_systems.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    run_library_benchmark(seeds=seeds, campaigns=tuple(camps), verbose=verbose, out_name="more_systems_library.json")
    return out
