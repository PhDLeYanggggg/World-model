import numpy as np
import pytest
import torch
from src.world_model import m3w_aux_prior as m
from src.world_model import m3w_strong_cap_auxiliary as old
from tests.test_m3w_strong_cap_auxiliary import data


def run_fit(tmp, module, arm='cap_aux', **kwargs):
    torch.set_num_threads(4)
    x, y, e, t, s, d, pr, cfg = data()
    model, fit = module.fit(x, y, e, t, s, 'outside', d, pr, arm=arm,
        seed=17, settings=cfg, identity={'test': 1}, directory=tmp,
        heartbeat=lambda **_: None, **kwargs)
    return model, module.restore(tmp)[1], fit


@pytest.mark.parametrize('arm', old.ARMS)
def test_legacy_numerical_identity(tmp_path, arm):
    _, a, _ = run_fit(tmp_path/'a', old, arm)
    _, b, _ = run_fit(tmp_path/'b', m, arm, initialization='inherited_easy')
    for group in ('initial_model', 'model'):
        for key in a[group]: assert torch.equal(a[group][key], b[group][key])
    for key in ('draws', 'fixed_ids', 'loss_scales', 'auxiliary_target'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('sampler_rng', 'torch_rng'): assert torch.equal(a[key], b[key])
    assert a['trace'] == b['trace']
    for k, state in a['optimizer']['state'].items():
        for name, v in state.items(): assert torch.equal(v, b['optimizer']['state'][k][name])


@pytest.mark.parametrize('arm', ['cap_aux', 'shuffled_aux'])
def test_only_intercept_changes_and_resume(tmp_path, arm):
    _, old_state, _ = run_fit(tmp_path/'old', old, arm)
    _, a, fit = run_fit(tmp_path/'a', m, arm)
    run_fit(tmp_path/'b', m, arm, stop_at=4)
    _, b, _ = run_fit(tmp_path/'b', m, arm, resume=True)
    for key in a['initial_model']:
        if key != 'membership.bias': assert torch.equal(a['initial_model'][key], old_state['initial_model'][key])
    for key in a['model']: assert torch.equal(a['model'][key], b['model'][key])
    np.testing.assert_array_equal(a['draws'], old_state['draws'])
    assert fit['unknown_rows_sampled'] == 0
    with pytest.raises(ValueError, match='initialization'):
        run_fit(tmp_path/'b', m, arm, resume=True, initialization='inherited_easy')


def test_prior_masks_and_edge_support():
    assert m.fitting_prior(np.array([0., 1., np.nan]), np.array([.2, .3, .5])) == .6
    assert m.fitting_prior(np.array([0., 0.]), np.ones(2)) == 0
    assert m.fitting_prior(np.array([1., 1.]), np.ones(2)) == 1
    with pytest.raises(ValueError): m.fitting_prior(np.array([np.nan]), np.ones(1))
    with pytest.raises(ValueError): m.fitting_prior(np.array([.1]), np.ones(1))


def test_matched_true_shuffled_initialization(tmp_path):
    _, a, _ = run_fit(tmp_path/'a', m, 'cap_aux')
    _, b, _ = run_fit(tmp_path/'b', m, 'shuffled_aux')
    assert a['cap_prior'] == b['cap_prior']
    for key in a['initial_model']: assert torch.equal(a['initial_model'][key], b['initial_model'][key])
    assert m.fitting_prior(a['auxiliary_target'], a['preprocess']['weights']) == pytest.approx(
        m.fitting_prior(b['auxiliary_target'], b['preprocess']['weights']))
