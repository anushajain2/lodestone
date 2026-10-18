import pytest
from pydantic import ValidationError

from lodestone.data.catalog import load_catalog
from lodestone.schemas import ExperimentResult, TargetSpec
from lodestone.targets import explain_exclusions, filter_pool, parse_target_text


def test_target_validation():
    TargetSpec(k_min=15, gap_min_eV=4)
    with pytest.raises(ValidationError):
        TargetSpec(k_min=30, k_max=10)
    with pytest.raises(ValidationError):
        TargetSpec(excluded_elements=["Xx"])
    with pytest.raises(ValidationError):
        TargetSpec(excluded_elements=["O"], required_elements=["O"])
    with pytest.raises(ValidationError):
        TargetSpec(budget=10, initial_batch=20)


def test_filter_pool_respects_constraints():
    cat = load_catalog()
    spec = TargetSpec(required_elements=["O"], max_elements=2, e_hull_max_eV_per_atom=0.0)
    pool = filter_pool(cat, spec)
    assert len(pool) > 10
    assert all("O" in els and len(els) <= 2 for els in pool["elements"])
    assert (pool["e_hull"] <= 1e-12).all()
    assert not any(set(els) & {"Pb", "Cd"} for els in pool["elements"])
    ex = explain_exclusions(cat, spec)
    assert ex["allowed"] == len(pool)


@pytest.mark.parametrize(
    "text,expect",
    [
        ("k above 15, gap above 4 eV, no lead", {"k_min": 15, "gap_min_eV": 4}),
        ("dielectric constant at least 20, gap above 4.5 eV, e_hull below 25 meV, no Hf", {"k_min": 20, "gap_min_eV": 4.5, "e_hull_max_eV_per_atom": 0.025}),
        ("k between 10 and 30, gap greater than 6 eV, binary compounds only", {"k_min": 10, "k_max": 30, "gap_min_eV": 6}),
        ("Find nitrides with k > 8 and Eg > 3 eV within 30 experiments", {"k_min": 8, "gap_min_eV": 3, "budget": 30}),
    ],
)
def test_plain_english(text, expect):
    spec, notes = parse_target_text(text)
    for k, v in expect.items():
        assert getattr(spec, k) == pytest.approx(v), (k, spec)
    assert notes


def test_plain_english_exclusions_and_anions():
    spec, _ = parse_target_text("high-k oxide, no lead or cadmium, avoid rare earths")
    assert {"Pb", "Cd", "La", "Y"} <= set(spec.excluded_elements)
    assert spec.required_elements == ["O"]
    assert spec.k_min == 15


def _result(**over):
    base = dict(
        result_id="r", request_id="q", partner_id="p", program_id="g", material_id="mp-352", formula="HfO2", status="ok",
        dielectric_constant={"value": 20, "unit": "dimensionless", "method": "CV"},
        band_gap={"value": 5.5, "unit": "eV", "method": "SE"},
        provenance={"lab_id": "helios", "lab_name": "H", "instrument": "x", "measured_day": 1, "received_day": 1, "raw_file": "f", "adapter": "a"},
    )
    base.update(over)
    return base


def test_result_contract():
    ExperimentResult(**_result())
    with pytest.raises(ValidationError):
        ExperimentResult(**_result(band_gap={"value": 5500, "unit": "meV", "method": "SE"}))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result(dielectric_constant={"value": -3, "unit": "dimensionless", "method": "CV"}))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result(status="failed", dielectric_constant=None, band_gap=None))  # failure needs a reason
    with pytest.raises(ValidationError):
        ExperimentResult(**_result(extra_field=1))
