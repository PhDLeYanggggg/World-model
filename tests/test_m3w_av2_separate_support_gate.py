import pytest

from scripts import run_m3w_av2_separate_support_gate as study


def row(score, *, eligible, support):
    return dict(scores=dict(conditional_harm=score, gated_harm=score if support else None),
                result=dict(eligible=eligible, risk=eligible, gain=(1., 1.)),
                support=support, harm=support and not eligible)


def test_support_gate_is_noncompensating_and_inclusive():
    raw = dict(actor_support=2., conditional_harm=1e9)
    assert all(study.gate(raw, .49)[k] is None for k in study.GATED)
    assert study.gate(raw, .50) == dict(raw, gated_support=2., gated_harm=1e9, high_support_harm=None)
    assert study.gate(raw, .75)["high_support_harm"] == raw["conditional_harm"]
    with pytest.raises(ValueError):
        study.gate(raw, float("nan"))


def test_calibration_checks_admitted_risk_and_rejects_empty():
    admitted = [row(3., eligible=True, support=True), row(2., eligible=False, support=True)]
    unsupported = row(4., eligible=False, support=False)
    assert study.calibrate([unsupported, *admitted], "gated_harm")["accepted"] == 1
    assert study.calibrate([unsupported, *admitted], "conditional_harm")["accepted"] == 0
    assert study.calibrate([unsupported], "gated_harm")["threshold"] is None


def test_common_count_reduces_budget_without_spill():
    arms = ("conditional_harm", "gated_harm")
    unsupported = row(4., eligible=False, support=False)
    admitted = row(3., eligible=True, support=True)
    matched = study.matched_rows([unsupported, admitted], arms, "city")
    assert matched["n"] == 1
    assert matched["arms"]["conditional_harm"]["unsupported"] == 1
    assert matched["arms"]["gated_harm"]["qualified"] == 1
    assert study.matched_rows([unsupported], arms, "city") is None
