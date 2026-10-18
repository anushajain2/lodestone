"""Stability re-check with an open machine-learned force field (C3): MACE-MP-0 through ASE.

For a shortlisted material: relax the Materials Project structure with MACE-MP-0 and report how
far it moves. A structure that is a true local minimum barely changes; large energy drops or volume
changes flag that the DFT structure (or the MLFF) disagrees, so a human should look before a lab
spends time on it. This is a sanity check on a shortlist, not a replacement for a convex hull.

MACE-MP-0 weights download on first use (~20 MB).
Results are cached in var/stability_cache.json.
"""

from __future__ import annotations

import json
import time
import warnings
from functools import lru_cache

from lodestone import config
from lodestone.data.catalog import structure_of

CACHE = config.VAR_DIR / "stability_cache.json"


def available() -> bool:
    try:
        import mace  # noqa: F401
        import ase  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _calculator():
    from mace.calculators import mace_mp

    warnings.filterwarnings("ignore")
    return mace_mp(model="small", default_dtype="float64", device="cpu")


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def recheck(material_id: str, fmax: float = 0.05, steps: int = 150) -> dict:
    cache = _load_cache()
    if material_id in cache:
        return {**cache[material_id], "cached": True}
    if not available():
        return {"status": "unavailable", "detail": "run `uv sync` to install mace-torch"}
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    t0 = time.time()
    s = structure_of(material_id)
    if len(s) > 40:
        return {"status": "skipped", "detail": f"{len(s)} atoms; limit is 40 for a laptop re-check"}
    atoms = AseAtomsAdaptor.get_atoms(s)
    atoms.calc = _calculator()
    n = len(atoms)
    e0 = atoms.get_potential_energy() / n
    f0 = float((atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5)
    v0 = atoms.get_volume()
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)
    e1 = atoms.get_potential_energy() / n
    dv = (atoms.get_volume() - v0) / v0
    flag = (e0 - e1) > 0.05 or abs(dv) > 0.05
    out = {
        "status": "ok",
        "model": "MACE-MP-0 (small)",
        "max_force_initial_eV_per_A": round(f0, 3),
        "relaxation_energy_eV_per_atom": round(e0 - e1, 4),
        "volume_change_pct": round(float(100 * dv), 2),
        "converged": bool(converged),
        "verdict": "check structure" if flag else "consistent with DFT structure",
        "seconds": round(time.time() - t0, 1),
    }
    cache[material_id] = out
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    return out
