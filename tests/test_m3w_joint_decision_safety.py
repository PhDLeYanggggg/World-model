import numpy as np
import pytest
import torch
from src.world_model import m3w_joint_decision_risk as api
from tests.test_m3w_decoupled_risk import prepared


def test_unknown_only_batch_adds_no_joint_supervision(tmp_path):
    data, _, _ = prepared(tmp_path); data['observed_future'][:] = False
    r = torch.ones(4, 7, 2, requires_grad=True); d = -r
    args = (data, np.arange(4), torch.tensor([0, 0, 1, 1]), 2)
    a = api.objective((r, d), *args, 'joint')
    b = api.parent.objective((r, d), *args)
    assert a['composed_observed_cost'] == 0
    assert torch.equal(a['total'], b['total'])


def test_incomplete_optimizer_update_cannot_replace_atomic_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path)
    kw = dict(settings=settings, identity={}, arm='joint'); path = tmp_path/'state.pt.gz'
    api.fit(data, warm, **kw, path=path, stop_at=2); saved = path.read_bytes()
    original = torch.optim.AdamW.step
    def interrupted(self, *args, **kwargs):
        with torch.no_grad():
            self.param_groups[0]['params'][0].add_(10)
        raise KeyboardInterrupt('incomplete update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.fit(data, warm, **kw, path=path, resume=True)
    assert path.read_bytes() == saved
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    a = api.fit(data, warm, **kw, path=path, resume=True)
    b = api.fit(data, warm, **kw, path=tmp_path/'direct.pt.gz')
    api.parent.exact_state(a, b)


def test_joint_cost_and_conjunction_sign_are_same_in_normalized_and_deployed_units():
    rng = np.random.default_rng(3)
    r, d = rng.normal(size=(2, 9, 7, 2))
    scaled_r = r*rng.uniform(.001, 1000, (9, 1, 2))
    deployed = api.parent.combined(scaled_r, d)
    np.testing.assert_array_equal((deployed <= 0).all(-1), (np.maximum(r, d) <= 0).all(-1))
