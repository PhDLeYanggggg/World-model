import pytest
from scripts.report_m3w_european_fixed_producer_roles import EXPECTED, POLICIES, COMPARISONS, easy_decomposition, no_promotion_gates


def test_registered_readout_not_replication_count():
    assert len(POLICIES) == 5 and len(COMPARISONS) == 3
    assert EXPECTED == dict(saved_decisions_verified=180, coordinate_arrays_verified=144, metric_reductions_verified=2376)


def test_floor4_attribution_keeps_bad_controller_increment():
    view = dict(easy_vs_CV={'by_scene': {'s': dict(rows=4, model_error=104., reference_error=100., gain_percent=-4.)}},
        ADE_vs_floor4={'easy': {'by_scene': {'s': dict(rows=4, model_error=104., reference_error=100.5)}}})
    r = easy_decomposition(view)['s']
    assert r['floor4_degradation_vs_CV'] == .5 and r['controller_added_degradation_pp'] == 3.5
    assert r['controller_created_violation'] and not r['already_bad_floor']


def fixture(easy=3., negative=1):
    return dict(policies={'producer_matched': {'safety': {'worst_positive_easy_degradation_percent': easy, 'zero_CV_harm_views': 0,
        'easy_defined_views': 36, 'easy_expected_views': 36}}},
        matched_vs_controls={c: {'ADE': {'all': {'negative_CI': negative, 'positive_CI': 10}}} for c in COMPARISONS})


def test_positive_branches_do_not_erase_negative_controls_or_bad_easy():
    g = no_promotion_gates(fixture())
    assert not g['matched_gain_consistent_across_controls'] and not g['matched_easy_preserved']
    assert not g['deployment_promoted'] and not g['submission_ready']


def test_missing_easy_does_not_pass():
    assert not no_promotion_gates(fixture(easy=None))['matched_easy_preserved']


def test_one_missing_easy_view_cannot_be_hidden_by_other_safe_views():
    s = fixture(easy=1., negative=0)
    s['policies']['producer_matched']['safety']['easy_defined_views'] = 35
    assert not no_promotion_gates(s)['matched_easy_preserved']


def test_development_checks_cannot_unlock_confirmation_or_execution():
    g = no_promotion_gates(fixture(easy=1., negative=0))
    assert g['matched_gain_consistent_across_controls'] and g['matched_easy_preserved']
    assert not any(g[k] for k in ('independent_confirmation', 'deployment_promoted', 'submission_ready', 'stage5c_executed', 'smc_enabled'))
