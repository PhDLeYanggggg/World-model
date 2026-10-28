import inspect

import pytest

from scripts import read_m3w_task_risk_final_heads as heads
from scripts import run_m3w_task_risk_readout as run
from src.world_model import m3w_task_risk_training as fitting


def test_reader_rejects_unregistered_payload(tmp_path):
    with heads.Reader(dict(checkpoints={})) as reader:
        with pytest.raises(KeyError):reader.load(tmp_path/'unregistered.pt.gz')


def test_seven_arms_and_six_primary_contrasts_are_fixed():
    assert len(run.api.ARMS)==7
    assert fitting.ARMS==('quadratic','task_utility','task_risk')
    assert run.api.PRIMARY_COMPARATORS==('quadratic','task_utility','original')
    assert run.api.TARGET=='task_risk'


def test_readout_requires_full_freeze_and_never_calls_fit():
    source=inspect.getsource(run.run)
    assert '.fit(' not in source
    assert 'local_neural_checkpoint_bytes_read' in source
    assert 'training_diagnostics' in source
    assert 'fitting.assert_matched(trained)' in source
    assert "objective_config=cfg['objective']" in source
    assert 'new_training=False' in source and 'independent_roles_read=False' in source
    assert '216' in inspect.getsource(heads.local_admission)
