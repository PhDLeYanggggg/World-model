import pytest
from scripts.report_m3w_european_producer_conditioned import EXPECTED, POLICIES, COMPARISONS, safety, reliability_summary


def view(easy=-1., zero=0, ratio=.03, predicted=.01):
    return dict(easy_vs_CV=dict(worst_scene_gain_percent=easy), zero_CV=dict(rows=2, harmed_rows=zero),
        switch_rate=.1, unknown_ADE_switches=3,
        reliability={'a': {'selected': dict(rows=10, realized_harm_ratio=ratio, predicted_harm_ratio=predicted)}})


def test_complete_registered_matrix():
    assert len(POLICIES) == 5 and len(COMPARISONS) == 4
    assert EXPECTED == dict(saved_decisions_verified=180, coordinate_arrays_verified=216,
        metric_reductions_verified=3402, old_anchor_metrics_exact=90)


def test_safety_retains_bad_locality_and_zero_reference():
    r = safety([view(easy=.5), view(easy=-3., zero=1)])
    assert r['worst_positive_easy_degradation_percent'] == 3.
    assert r['zero_CV_harm_views'] == 1
    assert r['unknown_ADE_switches_range'] == [3, 3]


def test_easy_improvement_not_negative_degradation():
    assert safety([view(easy=2.)])['worst_positive_easy_degradation_percent'] == 0.


def test_missing_easy_not_safe_by_default():
    assert safety([view(easy=None)])['worst_positive_easy_degradation_percent'] is None


def test_harm_ratio_not_net_gain_and_counts_not_independent():
    r = reliability_summary([view(), view(ratio=.005), view(ratio=None, predicted=None)])
    assert r['dependent_locality_views'] == 3 and r['selected_supported_views'] == 2
    assert r['realized_above_2pct'] == 1 and r['underpredicted_views'] == 1
    assert r['realized_harm_ratio_range'] == pytest.approx([.005, .03])
