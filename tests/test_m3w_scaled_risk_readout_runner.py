import inspect

import pytest

from scripts import read_m3w_scaled_risk_heads as heads
from scripts import run_m3w_scaled_risk_readout as run


def test_reader_rejects_unregistered_payload(tmp_path):
    with heads.Reader(dict(checkpoints={})) as reader:
        with pytest.raises(KeyError):reader.load(tmp_path/'unknown.pt.gz')


def test_four_primary_comparators_are_fixed():
    assert len(run.api.ARMS)==8
    assert run.api.PRIMARY_COMPARATORS==('unit_control','quadratic','task_utility','original')
    assert run.api.TARGET=='budget_scaled'
    assert heads.NEURAL_ARMS==('quadratic','task_utility','unit_control','budget_scaled')


def test_full_freeze_and_control_replay_not_refitting():
    source=inspect.getsource(run.run)
    assert '.fit(' not in source
    assert 'scaling.assert_matched(trained[-2:])' in source
    assert "prior['prediction_hashes'][old_name]" in source
    assert 'cached_neural_control_predictions_exact=True' in source
    assert 'new_training=False' in source and 'independent_roles_read=False' in source
    assert 'api.assert_control' in inspect.getsource(heads.admission)
    assert '144new plus144cached controls required' in inspect.getsource(heads.admission)
