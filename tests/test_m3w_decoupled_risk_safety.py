import json
import numpy as np
import pytest
import torch
from src.world_model import m3w_decoupled_risk as api
from tests.test_m3w_decoupled_risk import prepared


def test_incomplete_optimizer_update_cannot_replace_resume_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1); d, warm, settings = prepared(tmp_path)
    kw = dict(arm='separate64', settings=settings, identity={})
    path = tmp_path/'state.pt.gz'
    api.fit(d, warm, **kw, path=path, stop_at=2)
    before = path.read_bytes()
    original_step = torch.optim.AdamW.step

    def interrupted_step(self, *args, **kwargs):
        with torch.no_grad():
            self.param_groups[0]['params'][0].add_(1)
        raise KeyboardInterrupt('simulated incomplete parameter update')

    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupted_step)
    with pytest.raises(KeyboardInterrupt):
        api.fit(d, warm, **kw, path=path, resume=True)
    assert path.read_bytes() == before
    monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    a = api.fit(d, warm, **kw, path=path, resume=True)
    b = api.fit(d, warm, **kw, path=tmp_path/'complete.pt.gz')
    api.exact_state(a, b)


def test_changed_outcome_or_observation_mask_cannot_reuse_frozen_origin(tmp_path):
    d, warm, _ = prepared(tmp_path)
    d['observed_future'] = ~d['observed_future']
    with pytest.raises(AssertionError):
        api.validate(d, warm)


def test_sign_gate_does_not_depend_on_incompatible_score_magnitudes():
    risk = np.array([[[-.01, -.05], [.2, -.02]], [[-.3, -.2], [-.4, -.8]]])
    logits = np.array([[[100., -100.], [-100., -2.]], [[-100., 5.], [-2., -2.]]])
    expected = (risk <= 0) & (logits <= 0)
    for scale in (1e-9, 1., 1e9):
        assert np.array_equal(api.combined(scale*risk, logits) <= 0, expected)


def test_failed_fitting_screen_never_fetches_held_packet(tmp_path, monkeypatch):
    from scripts import run_m3w_moving_decoupled_risk as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(complete_heads=9, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not fetch a held packet after failed expansion screen')
    monkeypatch.setattr(run.old, 'packet', forbidden)
    run.evaluate({}, None, lambda **kw: None, lambda: None)
    result = json.loads((tmp_path/'analysis.json').read_text())
    assert result['status'] == 'fitting_complete_held_readout_not_run'
    assert not result['identity_certification'] and result['original_rules_unchanged']
