"""Export a program's results in an OPTIMADE-compatible form (X3).

Each measured material becomes an OPTIMADE ``structures`` entry (standard fields from the
structure) with the lab measurements under provider-prefixed fields, as the OPTIMADE spec requires
for non-standard properties (prefix ``_lodestone_``).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pymatgen.core import Composition

from lodestone.data.catalog import load_catalog, structure_of
from lodestone.store import PartnerScope

PREFIX = "_lodestone_"


def _entry(material_id: str, measurements: list[dict]) -> dict:
    s = structure_of(material_id)
    comp: Composition = s.composition
    elements = sorted(e.symbol for e in comp.elements)
    red = comp.reduced_composition
    ratios = [comp.get_atomic_fraction(e) for e in elements]
    species = sorted({site.specie.symbol for site in s})
    k_vals = [m["k"] for m in measurements]
    g_vals = [m["gap"] for m in measurements]
    return {
        "id": material_id,
        "type": "structures",
        "attributes": {
            "elements": elements,
            "nelements": len(elements),
            "elements_ratios": ratios,
            "chemical_formula_reduced": "".join(
                f"{e}{'' if int(red[e]) == 1 else int(red[e])}" for e in elements
            ),
            "chemical_formula_descriptive": comp.reduced_formula,
            "nsites": len(s),
            "lattice_vectors": s.lattice.matrix.round(6).tolist(),
            "cartesian_site_positions": [list(map(lambda x: round(float(x), 6), site.coords)) for site in s],
            "species_at_sites": [site.specie.symbol for site in s],
            "species": [{"name": sp, "chemical_symbols": [sp], "concentration": [1.0]} for sp in species],
            "dimension_types": [1, 1, 1],
            "nperiodic_dimensions": 3,
            "structure_features": [],
            "last_modified": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            f"{PREFIX}dielectric_constant": round(sum(k_vals) / len(k_vals), 3),
            f"{PREFIX}band_gap_eV": round(sum(g_vals) / len(g_vals), 3),
            f"{PREFIX}n_measurements": len(measurements),
            f"{PREFIX}measurements": measurements,
            f"{PREFIX}e_hull_eV_per_atom": float(load_catalog().loc[material_id, "e_hull"]),
        },
    }


def export_program(scope: PartnerScope, program_id: str) -> dict:
    results = scope.list_results(program_id, status="ok")
    by_mat: dict[str, list[dict]] = {}
    for r in results:
        prov = r["result"]["provenance"]
        by_mat.setdefault(r["material_id"], []).append(
            {"k": r["k"], "gap": r["gap"], "lab_id": prov["lab_id"], "method": prov["instrument"], "day": prov["received_day"], "result_id": r["result_id"]}
        )
    scope.log("export", "program", program_id, detail=f"OPTIMADE, {len(by_mat)} entries")
    return {
        "meta": {
            "api_version": "1.2.0",
            "provider": {"prefix": "lodestone", "name": "Lodestone (prototype)", "description": "Simulated-lab measurements; structures from Materials Project"},
            "data_returned": len(by_mat),
            "more_data_available": False,
            "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "data": [_entry(m, ms) for m, ms in sorted(by_mat.items())],
    }
