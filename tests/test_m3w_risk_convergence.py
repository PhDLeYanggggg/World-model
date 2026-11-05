import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_risk_convergence as m


def fixture(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(120, 120))
    data = dict(x=x, target=x[:, :2]*.02, weights=np.ones(120),
                sites=np.array(['fit']*100+['held']*20), ids=np.arange(120))
    parent = m.parent.fit(data, 'held', m.parent.SETTINGS, {'test': True}, tmp_path/'parent.pt.gz')
    return data, parent


def test_continuation_resume_preserves_optimizer_and_parent(tmp_path):
    d, parent = fixture(tmp_path)
    unchanged = copy.deepcopy(parent)
    a = m.continue_fit(d, 'held', parent, {}, tmp_path/'a.pt.gz', stop_extra_steps=8)
    m.continue_fit(d, 'held', parent, {}, tmp_path/'b.pt.gz', stop_extra_steps=3)
    b = m.continue_fit(d, 'held', parent, {}, tmp_path/'b.pt.gz', resume=True, stop_extra_steps=8)
    m.exact_state(a, b)
    m.parent.storage.base.exact(parent, unchanged)
    assert a['step'] == parent['step']+8 and not a['complete']
    assert a['target_step']-a['start_step'] >= 4096


def test_held_labels_and_features_cannot_change_fitting(tmp_path):
    d, parent = fixture(tmp_path)
    a = m.continue_fit(d, 'held', parent, {}, tmp_path/'a.pt.gz', stop_extra_steps=4)
    d['target'][100:] = 1e7
    d['x'][100:] = -1e7
    b = m.continue_fit(d, 'held', parent, {}, tmp_path/'b.pt.gz', stop_extra_steps=4)
    m.exact_state(a, b)


def test_partial_parent_rejected(tmp_path):
    d, parent = fixture(tmp_path)
    parent['complete'] = False
    with pytest.raises(ValueError, match='Complete registered'):
        m.continue_fit(d, 'held', parent, {}, tmp_path/'bad.pt.gz')


def test_loss_plateau_is_not_convergence_or_safety():
    state = dict(extra_trace=[dict(extra_steps=0, weighted_fit_MSE=1.),
                             dict(extra_steps=3072, weighted_fit_MSE=.5),
                             dict(extra_steps=4096, weighted_fit_MSE=.499)])
    result = m.fitting_diagnosis(state)
    assert result['operational_plateau']
    assert not result['mathematical_convergence_proved']
    assert not result['selected_by_held_outcomes']
