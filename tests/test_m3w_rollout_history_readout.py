import json
import pytest
from scripts import run_m3w_moving_rollout_history as run


def test_failed_fitting_screen_does_not_open_new_held_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps({'complete_heads': 6, 'all_before_validation': True}))
    (tmp_path/'fitting_screen.json').write_text(json.dumps({'followup_readout_allowed': False}))
    def forbidden(*a, **kw):
        raise AssertionError('No held readout after failed fitting screen')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw: None, lambda: None)
    a = json.loads((tmp_path/'analysis.json').read_text())
    assert a['status'] == 'fitting_complete_held_readout_not_run'
    assert a['original_rules_unchanged'] and not a['identity_certification']


def test_incomplete_training_cannot_open_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps({'complete_heads': 5, 'all_before_validation': True}))
    (tmp_path/'fitting_screen.json').write_text(json.dumps({'followup_readout_allowed': True}))
    with pytest.raises(ValueError, match='six'):
        run.evaluate({}, None, lambda **kw: None, lambda: None)
