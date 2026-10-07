import json

import pytest

from scripts import verify_m3w_supplement_v4 as api


@pytest.fixture(scope='module')
def result():
    return api.examples()


def test_loss_balances_queries_and_two_channel_groups(result):
    row = result['query_balanced_loss']
    assert row['scalar'] == pytest.approx(row['production'], abs=1e-14)
    assert row['row_weights'] == [.25, .25, .5]


def test_shared_unknowns_cancel_but_exchanged_unknowns_do_not(result):
    assert result['paired_unknown'] == dict(lower=-7., upper=7., lower_proxy_difference=-1., shared_cancelled=1)


def test_selected_risk_ratio_is_not_monotone(result):
    risk = result['threshold_counterexample']['easy_risks']
    assert risk == pytest.approx([.1, 1/1010, .05])
    assert risk[0] > .02 > risk[1] and risk[2] > .02


def test_temporal_positive_harm_is_not_original_harm(result):
    assert result['temporal_identity'] == dict(gross_harm=1., cancellation=.5, trajectory_harm=.5)


def test_deviance_log_gradient_motivation_not_real_improvement(result):
    assert result['log_gradient']['deviance'] == pytest.approx(-2, abs=1e-8)
    assert abs(result['log_gradient']['quadratic']) < 1e-8
    assert result['optimizer_updates'] == 0
    assert not result['scientific_result'] and not result['real_data_read']
    assert not result['population_safety_guarantee']


def test_export_records_source_bindings_and_no_new_data_claims():
    row = json.loads(api.artifact())
    assert len(row['source_sha256']) == 11
    assert row['result_source'] == 'fresh_run_synthetic_identity_checks_only'
    assert not row['independent_roles_read']
