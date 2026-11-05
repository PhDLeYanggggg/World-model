import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from src.world_model import m3w_kinematic_risk as api
from tests.test_m3w_direct_tail_risk import data


def fixture():
    d = data(); x = d['causal']['x']; x[:] = 0
    history = x[..., :24].reshape(4, 7, 8, 3)
    history[..., 0] = np.arange(-7, 1)
    history[..., 2] = np.arange(-7, 1)/12
    x[..., 24:32] = 1
    for start in (306, 330, 355):
        x[..., start:start+24].reshape(4, 7, 12, 2)[..., 0] = np.arange(1, 13)
    x[..., 330:354] *= np.arange(1, 8)[None, :, None]/7
    d['z'] = torch.from_numpy(x.copy()); d['z'][..., 380:] = 0
    return d


def test_named_motion_basis_and_causal_neighbor_mask():
    d = fixture(); x = d['causal']['x']; b = api.motion_basis(x)
    assert b.shape == (4, 7, 64) and len(set(api.NAMES)) == 64
    np.testing.assert_allclose(b[..., 14:21], 1)
    np.testing.assert_allclose(b[..., 21:27], 0)
    np.testing.assert_allclose(b[..., 39:42], np.broadcast_to([7, 7, 1], (4, 7, 3)))
    assert not b[..., 42:50].any()
    changed = x.copy(); changed[..., 32:224] = 999
    np.testing.assert_array_equal(b, api.motion_basis(changed))
    n = x[..., 32:224].reshape(4, 7, 8, 8, 3)
    n[..., 0, :, 0] = 2
    x[..., 224:288].reshape(4, 7, 8, 8)[..., 0, :] = 1
    b = api.motion_basis(x)
    np.testing.assert_allclose(b[..., 42:50], np.broadcast_to([1/8, 2, 2, 1, 1, 2/12, 0, 1], (4, 7, 8)), rtol=1e-6)
    with pytest.raises(ValueError, match='binary'):
        x[..., 224] = .5; api.motion_basis(x)


def test_warm_identity_and_matched_real_updates_resume(tmp_path):
    torch.set_num_threads(1); d = fixture()
    settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=4,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    warm = api.parent.fit(d, arm='mean', seed=17, settings=settings, identity={}, path=tmp_path/'warm.pt.gz')
    prepared = api.prepare(d, warm); m = api.initialize(warm)
    original = api.parent.DirectRiskHead(428, 4, 17); original.load_state_dict(warm['model'])
    seg = torch.tensor([0, 0, 1, 1])
    for arm in api.ARMS:
        assert torch.equal(m(api.batch_features(prepared, arm), seg, 2), original(d['z'], seg, 2))
    states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); args = dict(arm=arm, settings=settings, identity={'fixed': True})
        api.fit(prepared, warm, **args, path=path, stop_at=2)
        a = api.fit(prepared, warm, **args, path=path, resume=True)
        b = api.fit(prepared, warm, **args, path=tmp_path/(arm+'_repeat.pt.gz'))
        api.parent.parent.router.exact_state(a, b)
        np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        states.append(a)
    api.assert_matched(*states)
    assert not states[0]['model']['kinematics.weight'].any()
    assert states[1]['model']['kinematics.weight'].any()
    assert not torch.equal(states[0]['model']['encoder.weight'], warm['model']['encoder.weight'])
    before = api.predict(states[1], d['causal'])
    d['truth_by_arm']['bound_aware'][:] = 1e6; d['observed_future'][:] = False
    np.testing.assert_array_equal(before, api.predict(states[1], d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(states[1], dict(d['causal'], future_endpoint=np.zeros((4, 2))))
    with pytest.raises(AssertionError):
        api.prepare(d, warm)


def test_zero_motion_finite_and_query_isolation():
    d = fixture(); d['causal']['x'][..., :24] = 0
    assert np.isfinite(api.motion_basis(d['causal']['x'])).all()
    model = api.KinematicRiskHead(4, 17); torch.nn.init.normal_(model.excess.weight)
    x = torch.zeros(4, 7, 492); seg = torch.tensor([0, 0, 1, 1])
    before = model(x, seg, 2); x[2:] = 900
    assert torch.equal(before[:2], model(x, seg, 2)[:2])


def test_registered_trial_original_rules_and_outcome_order():
    from scripts import run_m3w_moving_kinematic_risk as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['primary_updates'] == 12000 and cfg['new_heads'] == 6
    assert cfg['risk_budget'] == cfg['easy_net_budget'] == .02
    assert not cfg['threshold_search'] and not cfg['new_H_readout'] and not cfg['identity_certification']
    tree = ast.parse(Path(run.__file__).read_text())
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert functions['evaluate'].index('api.parent.choose(') < functions['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 6" in functions['evaluate']


def test_extra_preprocessing_source_exclusion():
    from scripts import run_m3w_moving_kinematic_risk as run
    c = fixture()['causal']
    state = dict(fit_ids=np.array([999]), source_sites=['other'],
        preprocess=dict(source_names=['other']), extra_preprocess=dict(source_names=['other']))
    run.assert_excluded(state, c, ['s'])
    with pytest.raises(ValueError, match='preprocessing'):
        run.assert_excluded(dict(state, extra_preprocess=dict(source_names=['s'])), c, ['s'])
