import pytest

from lodestone.connectors import htem
from lodestone.connectors.adapters import HTEMAdapter, LabCSVAdapter, cation_fractions, to_partner_csv
from lodestone.evals.onboarding_eval import run_broken_records


def _lib(system="Co-Ni-O-Zn"):
    return next(lib for lib in htem.load_snapshot(system) if len(lib.get("samples") or []) >= 40)


def test_htem_records_have_units_and_provenance():
    lib = _lib()
    recs, rep = HTEMAdapter().adapt_library(lib)
    assert rep.received == len(lib["samples"]) == rep.accepted + rep.rejected
    r = recs[0]
    s = lib["samples"][0]
    assert r.measurements["conductivity"].unit == "S/cm" and r.measurements["band_gap"].unit == "eV"
    assert r.value("conductivity") == s["fpm_conductivity"] and r.value("band_gap") == s["opt_direct_bandgap"]
    assert r.provenance.source_sample_id == str(s["id"]) and r.provenance.source_library_id == str(lib["id"])
    assert r.provenance.url.endswith(f"/sample/{s['id']}")
    assert abs(sum(r.composition.values()) - 1) < 1e-9


def test_partner_csv_round_trip_converts_units():
    recs, _ = HTEMAdapter().adapt_library(_lib())
    text = to_partner_csv(recs[:10])
    assert "Rho [mOhm*cm]" in text and "Substrate T [K]" in text
    recs2, reps = LabCSVAdapter().adapt_csv(text, "Co-Ni-O-Zn")
    assert len(recs2) == 10 and reps[0].rejected == 0
    for a, b in zip(recs[:10], recs2):
        assert b.value("conductivity") == pytest.approx(a.value("conductivity"), rel=1e-4)
        assert b.value("band_gap") == pytest.approx(a.value("band_gap"), abs=2e-3)
        assert b.synthesis.deposition_temp_c == pytest.approx(a.synthesis.deposition_temp_c, abs=0.1)
        assert b.composition == pytest.approx(a.composition, abs=1e-3)


def test_every_broken_record_case_is_handled():
    rows = run_broken_records()
    assert len(rows) >= 15
    assert all(r["correct"] for r in rows), [r for r in rows if not r["correct"]]
    assert all(r["reason"] for r in rows if r["expected"] in ("rejected", "dropped_measurement"))


def test_cation_fractions_ignore_anions_and_validate():
    assert cation_fractions(["InO", "ZnO", "SnO2"], [50, 30, 20]) == pytest.approx({"In": 0.5, "Sn": 0.2, "Zn": 0.3})
    with pytest.raises(ValueError):
        cation_fractions(["Zn", "Sn"], [None, None])
    with pytest.raises(ValueError):
        cation_fractions(["Zn", "Sn"], [150, 100])


def test_import_into_workspace_reports_and_blocks_duplicates(seeded):
    from lodestone.onboarding import OnboardingError, import_htem_library

    db, ids, _ = seeded
    scope = db.scope("partner-a")
    lib = scope.get_library(ids["partner-a"]["library"])
    assert lib["report"]["received"] >= lib["accepted"]
    with pytest.raises(OnboardingError):
        import_htem_library(scope, int(lib["source_library_id"]))


def test_real_data_problems_are_rejected_with_reasons():
    """HTEM itself has ZnSnN samples with no XRF composition: they must be rejected, not guessed."""
    seen = set()
    rejected = 0
    for lib in htem.load_snapshot("N-Sn-Zn"):
        _, rep = HTEMAdapter().adapt_library(lib, seen)
        rejected += rep.rejected
        assert all(i.reason for i in rep.issues)
    assert rejected > 0
