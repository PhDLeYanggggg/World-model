import copy

import pytest

from scripts import report_m3w_decoder_risk_factorial as report
from tests.test_m3w_decoder_risk_readout import predictions
from tests.test_m3w_shared_train_risk import packet


def fixture():
    result, _ = report.evaluation.evaluate(packet(), predictions())
    row = dict(scope='held', result=result, outer_development_read=False,
               independent_roles_read=False, deployment_changed=False)
    partition = dict(held_recordings=list(result['recordings']), fit_recordings=['other'])
    return row, partition


def test_frozen_recording_partition_and_factorial_interaction():
    row, partition = fixture()
    report.check_view(row, partition)
    changed = copy.deepcopy(row)
    changed['result']['contrasts']['decoder_by_risk_MSE_interaction'] = 1.
    with pytest.raises(ValueError):
        report.check_view(changed, partition)
    with pytest.raises(ValueError):
        report.check_view(row, dict(partition, held_recordings=['unregistered']))


def test_new_data_role_cannot_be_hidden():
    row, partition = fixture()
    row['independent_roles_read'] = True
    with pytest.raises(ValueError):
        report.check_view(row, partition)


def test_missing_model_identity_rejected():
    reg = dict(contexts={'a': []}, seeds=[17])
    with pytest.raises(ValueError):
        report.check_roster(reg, {})


def test_report_preserves_undefined_and_negative_interval():
    assert report.interval(dict(CI95=None)) == 'not estimable'
    assert report.interval(dict(mean=-2., CI95=[-3., -1.])) == '-2.000000 [-3.000000, -1.000000]'
