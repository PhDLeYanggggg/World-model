import copy
import pytest
import numpy as np
from test_m3w_cost_support_diagnostic import fitted_fixture
from src.world_model import m3w_cost_support_diagnostic as api
from scripts import verify_m3w_cost_support_diagnostic as check
from scripts.run_m3w_cost_support_diagnostic import summarize


def report_fixture():
    (state, x, e, y, q, s, r, f), fitted, tables = fitted_fixture()
    old, new, _, desc, extra = api.raw_predictions(state, fitted, x, e, q, tables)
    raw = dict(original=old, cost=new)
    projected = {k: api.forest.project_moments(v, e) for k, v in raw.items()}
    actions = {k: np.zeros(len(y), bool) for k in ('original', 'cost', 'original_matched_cost', 'cost_matched_original')}
    d = api.diagnose(state, raw, projected, y, actions, s, r, f, desc, extra)
    score = api.tree_score_diagnostic(state, fitted, x, e, q, y, s, r, f)
    loss = fitted['training_loss']
    row = dict(source='source-a', checkpoint='checkpoint', training_ids_hash='train',
        validation_ids_hash='validation', targets_hash='target',
        prediction_hashes={k: k+'-prediction' for k in raw},
        action_hashes={k: k+'-action' for k in actions}, diagnosis=d,
        training_projected_error_change=d['cohorts']['all'], tree_scores=dict(train=score, validation=score),
        surrogate_reconstruction_residual=score['mean_tree_MSE_change']+score['fixed_penalty']-(sum(loss['after'])-sum(loss['before'])))
    anchor = {k: copy.deepcopy(row[k]) for k in ('checkpoint', 'training_ids_hash', 'validation_ids_hash',
                                               'targets_hash', 'prediction_hashes', 'action_hashes')}
    anchor['prediction_hashes']['other_control'] = 'untouched-control'
    anchor['result'] = dict(scores=dict(original=d['cohorts']['all']['global_weighted_old_MSE'],
        cost=d['cohorts']['all']['global_weighted_new_MSE']), policies={k: dict(selected_count=0) for k in raw},
        contrasts=dict(cost_minus_original_signed_MSE=d['cohorts']['all']['global_weighted_MSE_change']))
    anchor['training'] = dict(training_loss=loss)
    return row, anchor


def test_independent_reader_preserves_parent_subsets():
    row, anchor = report_fixture()
    assert check.verify_row(row, anchor) > 100


@pytest.mark.parametrize('corruption', ['action', 'decomposition', 'unknown', 'surrogate'])
def test_independent_reader_rejects_corrupted_fields(corruption):
    row, anchor = report_fixture()
    if corruption == 'action':
        row['action_hashes']['cost'] = 'changed'
    elif corruption == 'decomposition':
        row['diagnosis']['cohorts']['all']['moment_contributions'][4] += .01
    elif corruption == 'unknown':
        row['diagnosis']['cohorts']['all']['unknown_rows'] = 0
    else:
        row['surrogate_reconstruction_residual'] = .01
    with pytest.raises(AssertionError):
        check.verify_row(row, anchor)


def test_independent_bootstrap_matches_production_and_keeps_empty_support():
    a, _ = report_fixture()
    b = copy.deepcopy(a)
    b['source'] = 'source-b'
    rows = [a, copy.deepcopy(a), b]
    cfg = dict(bootstrap_resamples=101, bootstrap_seed=44)
    result = check.aggregate(rows, cfg)
    assert check.compare(result, summarize(rows, cfg)) > 100
    assert result['cohorts']['all']['global_weighted_MSE_change']['localities'] == 2
    assert result['support_cohorts']['cost_selected']['descriptor_mean']['zero_train_harm_fraction']['mean'] is None
