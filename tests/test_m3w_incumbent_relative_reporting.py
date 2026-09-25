import copy
import pytest
from scripts.report_m3w_european_incumbent_relative import gates, POLICIES, CONTROLS


def summary():
    metric = dict(positive_CI=3, negative_CI=0, defined=36)
    return dict(policies={'incumbent_reference': dict(ADE_vs_old_stop4={'all': metric},
        safety=dict(worst_positive_easy_degradation_percent=1., easy_defined_views=36, zero_CV_harm_views=0))},
        incumbent_vs_controls={'floor_reference': {'ADE': {'all': copy.deepcopy(metric)}}})


def test_registered_eight_policies_and_four_paired_comparisons():
    assert len(POLICIES) == 8 and len(CONTROLS) == 4
    assert 'add_only' in POLICIES and 'remove_only' in POLICIES


def test_negative_view_prevents_consistent_gain_claim():
    s = summary(); s['policies']['incumbent_reference']['ADE_vs_old_stop4']['all']['negative_CI'] = 1
    assert not gates(s)['incremental_gain_consistent']


@pytest.mark.parametrize('value', [None, 2.001])
def test_undefined_or_bad_easy_fails(value):
    s = summary(); s['policies']['incumbent_reference']['safety']['worst_positive_easy_degradation_percent'] = value
    assert not gates(s)['easy_preservation']


def test_partial_easy_coverage_not_a_pass():
    s = summary(); s['policies']['incumbent_reference']['safety']['easy_defined_views'] = 35
    assert not gates(s)['easy_preservation']


def test_development_success_cannot_promote_or_unlock_execution():
    g = gates(summary())
    assert g['incremental_gain_consistent'] and g['easy_preservation']
    assert not any(g[k] for k in ('independent_confirmation', 'independent_calibration',
        'deployment_promoted', 'submission_ready', 'stage5c_executed', 'smc_enabled'))


def test_decision_phase_is_label_free():
    import ast
    from pathlib import Path
    tree = ast.parse(Path('scripts/run_m3w_european_incumbent_relative.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decide')
    constants = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not {'target_eval', 'baseline_ade', 'target', 'valid'} & constants
    assert {'history', 'old_stop', 'previous_matched'} <= constants
