"""Column mappings (bring your own CSV) and the replay savings report."""

import json
import re

import pytest

from lodestone import config
from lodestone.connectors.adapters import LabCSVAdapter, Reject
from lodestone.connectors.mapping import ColumnMapping, helix_mapping, load_mapping

HELIX_CSV = config.DATA_DIR / "partner_lab" / "helix_second_lab_export.csv"
HELIX_MAPPING = config.DATA_DIR / "partner_lab" / "helix_mapping.yaml"
TEMPLATE_CSV = config.DATA_DIR / "templates" / "export_template.csv"
TEMPLATE_MAPPING = config.DATA_DIR / "templates" / "mapping_template.yaml"


def _adapt(mapping: dict, text: str):
    recs, reps = LabCSVAdapter(ColumnMapping(**mapping)).adapt_csv(text)
    return recs, [i for r in reps for i in r.issues]


BASE = {"system": "In-O-Sn-Zn", "columns": {"sample_id": "id", "composition": {"In": "In", "Zn": "Zn"}}}


def test_helix_export_imports_through_its_mapping():
    recs, reps = LabCSVAdapter(load_mapping(HELIX_MAPPING)).adapt_csv(HELIX_CSV.read_text())
    assert len(recs) == 176 and sum(r.rejected for r in reps) == 4
    assert all(i.reason.startswith("duplicate sample id") for r in reps for i in r.issues)
    r = recs[0]  # PL-234890: 773.1 K, 1.3332 Pa, 4478.28 mOhm*cm, edge 336.55 nm, 107.4 nm, 45.7 %
    assert r.synthesis.deposition_temp_c == pytest.approx(499.95)
    assert r.synthesis.growth_pressure_mtorr == pytest.approx(10.0, abs=1e-3)
    assert r.value("conductivity") == pytest.approx(1 / 4.47828)
    assert r.value("band_gap") == pytest.approx(1239.84193 / 336.55)
    assert r.value("thickness") == pytest.approx(0.1074) and r.value("visible_transmittance") == pytest.approx(0.457)
    assert r.provenance.sample_date == "2010-08-02" and r.provenance.position == 1
    # the adapter's default is the same mapping
    assert LabCSVAdapter().mapping == helix_mapping()


def test_template_mapping_reads_template_csv():
    m = load_mapping(TEMPLATE_MAPPING)
    adapter = LabCSVAdapter(m)
    text = TEMPLATE_CSV.read_text()
    recs, reps = adapter.adapt_csv(text)
    assert len(recs) == 12 and reps[0].rejected == 0 and not reps[0].issues
    r = recs[0]  # S314866: In 45.69 Sn 18.82 Zn 35.48 at%, 25 (degC from the mapping), 10 mTorr, 2139 S/cm, 3.625 eV, 210 nm
    assert r.composition == pytest.approx({"In": 0.4569, "Sn": 0.1882, "Zn": 0.3548}, abs=1e-3)
    assert r.synthesis.deposition_temp_c == 25 and r.synthesis.growth_pressure_mtorr == pytest.approx(10.0)
    assert r.value("conductivity") == 2139 and r.value("band_gap") == 3.625 and r.value("thickness") == pytest.approx(0.21)
    assert r.missing["visible_transmittance"] == "not in this lab's export"
    _, _, ignored = adapter.plan(text.splitlines()[0].split(","))
    assert ignored == ["Notes"]


def test_units_from_mapping_and_headers_are_converted():
    m = BASE | {"columns": BASE["columns"] | {
        "deposition_temperature": {"column": "T", "unit": "degC"},
        "resistivity": "rho", "conductivity": "sigma", "band_gap": "Eg",
        "thickness": "d", "transmittance": {"column": "Tvis", "unit": "fraction"}}}
    text = "id,In,Zn,T,rho [Ohm*m],sigma (S/m),Eg [meV],d [um],Tvis\nA1,60,40,300,0.001,1000,3100,0.25,0.8\n"
    (r,), issues = _adapt(m, text)
    assert not issues
    assert r.synthesis.deposition_temp_c == 300
    assert r.value("conductivity") == pytest.approx(10.0)  # 1000 S/m = 10 S/cm, agrees with 1/(0.1 Ohm*cm)
    assert r.value("band_gap") == pytest.approx(3.1) and r.value("thickness") == 0.25 and r.value("visible_transmittance") == 0.8
    assert r.composition == pytest.approx({"In": 0.6, "Zn": 0.4})


@pytest.mark.parametrize("columns,row,reason", [
    ({"deposition_temperature": "T"}, "A1,60,40,300", "has no unit"),
    ({"deposition_temperature": {"column": "T [K]", "unit": "degC"}}, "A1,60,40,300", "but the mapping says"),
    ({"thickness": "d [blorps]"}, "A1,60,40,5", "unknown unit"),
    ({"pressure": "p [nm]"}, "A1,60,40,5", "wrong dimension"),
    ({"conductivity": "s [S/cm]"}, "A1,60,40,lots", "non-numeric"),
    ({}, "A1,60,,", "unreadable Zn amount"),
    ({}, "A1,6,4,", "sum to 10.0%"),
])
def test_bad_rows_are_rejected_with_a_reason(columns, row, reason):
    extra = [c if isinstance(c, str) else c["column"] for c in columns.values()] or ["x"]
    text = "id,In,Zn," + ",".join(extra) + "\n" + row + "\n"
    recs, issues = _adapt(BASE | {"columns": BASE["columns"] | columns}, text)
    assert recs == [] and len(issues) == 1 and issues[0].level == "rejected"
    assert reason in issues[0].reason, issues[0].reason


def test_impossible_values_are_dropped_not_the_row():
    m = BASE | {"columns": BASE["columns"] | {"band_gap": "Eg [eV]", "resistivity": "rho [mOhm*cm]"}}
    (r,), issues = _adapt(m, "id,In,Zn,Eg [eV],rho [mOhm*cm]\nA1,60,40,40,-3\n")
    assert {i.field for i in issues} == {"band_gap", "conductivity"} and all(i.level == "dropped_measurement" for i in issues)
    assert r.missing["band_gap"] == "dropped by validation"


def test_fraction_composition_single_column_and_day_first_dates():
    m = {"system": "Co-Ni-O-Zn", "columns": {"sample_id": "id", "composition": "comp", "date": "when"},
         "composition_unit": "fraction", "date_format": "%d/%m/%Y"}
    (r,), issues = _adapt(m, "id,comp,when\nX-1,Co:0.5;Zn:0.5,31/12/2019\n")
    assert r.composition == {"Co": 0.5, "Zn": 0.5} and r.provenance.sample_date == "2019-12-31" and not issues
    (r,), issues = _adapt(m, "id,comp,when\nX-1,Co:0.5;Zn:0.5,2019-12-31\n")
    assert r.provenance.sample_date is None and issues[0].level == "warning"


def test_mapping_errors_are_caught_up_front():
    with pytest.raises(Reject, match="missing required columns"):
        LabCSVAdapter(ColumnMapping(**BASE)).adapt_csv("id,In\nA,1\n")
    with pytest.raises(ValueError, match="unknown field"):
        ColumnMapping(**BASE | {"columns": BASE["columns"] | {"colour": "c"}})
    with pytest.raises(ValueError, match="not both"):
        ColumnMapping(**BASE | {"columns": BASE["columns"] | {"band_gap": "a", "optical_edge": "b"}})
    with pytest.raises(ValueError, match="sample id"):
        ColumnMapping(system="In-O-Sn-Zn", columns={"composition": "c"})
    with pytest.raises(Reject, match="which material system"):
        LabCSVAdapter(ColumnMapping(columns={"sample_id": "id", "composition": "c"})).adapt_csv("id,c\n")


def test_import_partner_csv_takes_a_mapping(tmp_path):
    from lodestone.onboarding import import_partner_csv
    from lodestone.store import Database

    db = Database(tmp_path / "m.db")
    db.platform().create_partner("Lab", "lab")
    scope = db.scope("lab")
    reps = import_partner_csv(scope, TEMPLATE_CSV.read_text(), None, "export_template.csv", load_mapping(TEMPLATE_MAPPING))
    assert reps[0].accepted == 12 and scope.systems() == {"In-O-Sn-Zn": 12}
    db.close()


# ----------------------------------------------------------------------------------- report


def test_verdict_states_weak_and_negative_results_plainly():
    from lodestone.replay_report import _verdict

    def m(med, found, runs=10):
        return {"experiments_to_top_median": med, "found_top_in_budget": found, "runs": runs}

    assert _verdict({"experiments": 20, "interval": [8, 30]}, m(25, 10), m(45, 9), 5, 60, 10)[0] == "positive"
    assert _verdict({"experiments": 3, "interval": [-4, 9]}, m(40, 10), m(43, 10), 5, 60, 10)[0] == "inconclusive"
    v, text, _ = _verdict({"experiments": -10, "interval": [-18, -2]}, m(50, 9), m(40, 10), 5, 60, 10)
    assert v == "negative" and "slower" in text
    v, text, notes = _verdict({"experiments": 0, "interval": [0, 0]}, m(61, 3), m(61, 2), 5, 60, 4)
    assert v == "weak" and "does not support a savings claim" in text
    assert any("count as 61" in n for n in notes) and any("Only 4 seeds" in n for n in notes)


def test_replay_report_command_on_helix_export(tmp_path):
    from typer.testing import CliRunner

    from lodestone.cli import app

    out = tmp_path / "report.html"
    res = CliRunner().invoke(app, [
        "replay-report", str(HELIX_CSV), "--mapping", str(HELIX_MAPPING), "--goal", "maximize conductivity with band gap >= 3 eV",
        "--cost-per-experiment", "1500", "--seeds", "3", "--budget", "20", "--out", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert "imported 176 of 180 rows" in res.output
    page = out.read_text()
    data = json.loads(re.search(r'<script type="application/json" id="replay-data">(.*?)</script>', page, re.S).group(1))
    assert data["import"]["accepted"] == 176 and data["import"]["rejected"] == 4
    assert data["pool"] == 176 and data["budget"] == 20 and data["seeds"] == 3
    assert set(data["methods"]) == {"bayes-gp", "random", "grid"} and all(len(m["per_seed"]) == 3 for m in data["methods"].values())
    assert data["verdict"] in {"positive", "inconclusive", "weak", "negative"}
    s = data["savings"]["random"]
    assert s["money"] == pytest.approx(s["experiments"] * 1500)
    assert s["bayes_faster"] + s["tied"] + s["bayes_slower"] == 3
    assert any("Only 3 seeds" in n for n in data["verdict_notes"])
    # one self-contained page: chart, import summary, caveats, no external resources
    assert "<svg" in page and "duplicate sample id" in page and "This is a replay, not a forecast" in page
    assert not re.search(r'(src|href)="(https?:)?//', page)


def test_replay_report_refuses_too_little_data(tmp_path):
    from lodestone.replay_report import ReplayError, run_replay_report

    with pytest.raises(ReplayError, match="at least 30"):
        run_replay_report(TEMPLATE_CSV, "maximize conductivity", TEMPLATE_MAPPING)
