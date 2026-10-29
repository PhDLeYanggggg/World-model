import copy

import numpy as np
import pytest

from scripts import evaluate_m3w_shared_train_risk as evaluation
from scripts import report_m3w_shared_train_risk as report
from tests.test_m3w_shared_train_risk import packet


def result():
    prediction = np.tile([.1, 0., 1., 1., 0.], (32, 1))
    return evaluation.evaluate(packet(), {arm: prediction.copy() for arm in evaluation.ARMS})[0]


def test_shared_report_checks_complete_aggregate():
    report.check_result(result())


def test_shared_report_rejects_corrupted_risk_mass():
    value = result()
    value['policies']['shared_source']['selected_known_harm_mass'] += 1
    with pytest.raises(ValueError):
        report.check_result(value)


def test_shared_report_rejects_unknown_risk_imputation():
    value = copy.deepcopy(result()['policies']['local'])
    value['selected_known_easy_reference_mass'] = 0
    value['known_easy_positive_risk'] = 0
    with pytest.raises(ValueError):
        report.check_policy(value)


def test_shared_report_rejects_changed_paired_contrast():
    value = result()
    value['contrasts']['shared_source_minus_local_full'] += 1
    with pytest.raises(ValueError):
        report.check_result(value)


def test_negative_known_easy_degradation_is_valid():
    y = np.array([[.1, 0., 1., 1., 0.]])
    value = evaluation.readout.policy_score(y, np.array([True]), np.array([.5]))
    assert value['easy_degradation_upper'] < 0
    report.check_policy(value)
