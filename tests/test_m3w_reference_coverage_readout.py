import copy
import numpy as np
import pytest

from scripts.report_m3w_dimensionless_harm_forecast import metrics
from src.evaluation.m3w_reference_harm_readout import harm_summary
from src.evaluation.m3w_reference_coverage_readout import replay_control_metrics, budget_counts


def test_exact_dual_comparator_and_population_replay():
    identity = dict(trial='single2_seed17', site='locality', endpoint='ADE', subset='all')
    before = dict(identity, metric=metrics(np.array([1., 2.]), np.array([2., 3.]),
                                         np.array([2., 2.]), np.array([3., 3.])))
    row = dict(identity, metric=dict(old_gain_CV=before['metric']['new_gain_CV'],
                                    old_gain_selected=before['metric']['new_gain_selected']),
        original_ADE_contrast=dict(old_gain_CV=before['metric']['old_gain_CV'],
                                  old_gain_selected=before['metric']['old_gain_selected']))
    assert replay_control_metrics([row], dict(rows=[before])) == 4
    bad = copy.deepcopy(row); bad['metric']['old_gain_CV'] += .1
    with pytest.raises(AssertionError):
        replay_control_metrics([bad], dict(rows=[before]))
    with pytest.raises(ValueError):
        replay_control_metrics([row, row], dict(rows=[before, before]))


def test_unrestricted_forecast_risk_keeps_empty_and_unknown_support():
    known = dict(arm='new', subset='all', reference='CV', **harm_summary([2., 1.], [1., 2.]))
    empty = dict(arm='new', subset='all', reference='CV', **harm_summary([], []))
    result = budget_counts([known, empty])['new/all/CV']
    assert result['known_fail'] == 1  # Net zero does not cancel positive harm.
    assert result['not_estimable'] == 1
    assert result['required_views'] == 2
    empty['known_budget_pass'] = True
    with pytest.raises(ValueError):
        budget_counts([empty])
