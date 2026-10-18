"""Lab onboarding: get a lab's data into a partner workspace as clean, provenance-tracked records.

    import_htem_library(scope, 9940)            one HTEM library (snapshot first, live API otherwise)
    import_htem_system(scope, "In-O-Sn-Zn")     every library of a system in the snapshot
    import_partner_csv(scope, text, system)     a lab's CSV export, read through a column mapping

Every call returns the import report(s): what was received, accepted, rejected and why.
"""

from __future__ import annotations

from lodestone.connectors import htem
from lodestone.connectors.adapters import HTEMAdapter, LabCSVAdapter, Reject
from lodestone.connectors.mapping import ColumnMapping
from lodestone.schemas import ImportReport
from lodestone.store import PartnerScope


class OnboardingError(ValueError):
    pass


def _htem_library_raw(library_id: int) -> dict:
    for system in htem.available_systems():
        for lib in htem.load_snapshot(system):
            if int(lib["id"]) == int(library_id):
                return lib
    return htem.fetch_library(int(library_id))  # live API (cached under data/raw/htem)


def import_htem_library(scope: PartnerScope, library_id: int) -> ImportReport:
    ref = f"htem:{int(library_id)}"
    if any(lib["library_ref"] == ref for lib in scope.list_libraries()):
        raise OnboardingError(f"HTEM library {library_id} is already imported in this workspace")
    try:
        raw = _htem_library_raw(library_id)
    except LookupError as e:
        raise OnboardingError(str(e)) from None
    adapter = HTEMAdapter()
    records, report = adapter.adapt_library(raw, seen_ids=scope.existing_source_ids())
    synthesis = adapter.synthesis(raw, []).model_dump()
    scope.add_library(ref, adapter.source, str(library_id), report.system, synthesis, report.model_dump(), [r.model_dump() for r in records])
    return report


def import_htem_system(scope: PartnerScope, system: str, max_libraries: int | None = None) -> list[ImportReport]:
    try:
        libs = htem.load_snapshot(system)
    except FileNotFoundError as e:
        raise OnboardingError(str(e)) from None
    have = {lib["library_ref"] for lib in scope.list_libraries()}
    out = []
    for lib in libs[: max_libraries or len(libs)]:
        if f"htem:{lib['id']}" in have:
            continue
        out.append(import_htem_library(scope, lib["id"]))
    return out


def import_partner_csv(scope: PartnerScope, text: str, system: str | None, filename: str = "export.csv",
                       mapping: ColumnMapping | None = None) -> list[ImportReport]:
    """Import a CSV export. ``mapping`` says which column holds what (default: the Helix demo layout);
    ``system`` may be None when the mapping names it."""
    adapter = LabCSVAdapter(mapping)
    system = system or adapter.mapping.system
    try:
        records, reports = adapter.adapt_csv(text, system, seen_ids=scope.existing_source_ids())
    except Reject as e:
        raise OnboardingError(f"{filename}: {e}") from None
    for rep in reports:
        recs = [r for r in records if r.provenance.source_library_id == rep.source_library_id]
        ref = f"csv:{rep.source_library_id}"
        if any(lib["library_ref"] == ref for lib in scope.list_libraries()):
            raise OnboardingError(f"library {rep.source_library_id} from {filename} is already imported")
        synth = recs[0].synthesis.model_dump() if recs else None
        scope.add_library(ref, adapter.source, rep.source_library_id, system, synth, rep.model_dump(), [r.model_dump() for r in recs])
    return reports
