import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_multiaction_cost as base
from src.world_model import m3w_scene_context_cost as api
from tests.test_m3w_multiaction_cost import sample


def scene_sample():
    x, envelope, y, scales, sites, data = sample()
    scene = np.random.default_rng(17).normal(size=(*x.shape[:2], 18)).astype(np.float32)
    pr = api.preprocessing(data['preprocess'], scene, envelope, y, scales, sites)
    data['preprocess'] = pr
    extra = np.clip((scene-pr['mean'][-18:])/pr['std'][-18:], -pr['clip'], pr['clip']).astype(np.float32)
    data['z'] = api.SceneCausalFeatures(data['z'], torch.from_numpy(extra))
    return x, scene, envelope, data


SETTINGS = dict(width=8, steps=4, batch_size=8, learning_rate=.001,
    gradient_clip=5., heartbeat_every=2, checkpoint_every=2, underharm_weight=4., ranking_weight=.5)


def test_disabled_branch_has_exact_legacy_fit_and_optimizer(tmp_path):
    torch.set_num_threads(1); *_, original = sample()
    *_, padded = scene_sample()
    kw = dict(arm='physical_neural', settings=SETTINGS, seed=17, identity={'fixture':True})
    old = base.fit(original, **kw, path=tmp_path/'old.gz')
    with api.installed(False):
        new = base.fit(padded, **kw, path=tmp_path/'new.gz')
    api.assert_neutral_state(old, new)


def test_scene_recovery_all_banks_and_inactive_neural_isolation(tmp_path):
    torch.set_num_threads(1); x, scene, envelope, data = scene_sample(); states = []
    with api.installed():
        for arm in base.ARMS:
            kw = dict(data=data, arm=arm, settings=SETTINGS, seed=17, identity={'fixture':True})
            direct = base.fit(**kw, path=tmp_path/(arm+'.gz'))
            base.fit(**kw, path=tmp_path/(arm+'_resume.gz'), stop_at=2)
            resumed = base.fit(**kw, path=tmp_path/(arm+'_resume.gz'), resume=True)
            for key in direct:
                if key != 'seconds':
                    base.exact(direct[key], resumed[key])
            assert torch.count_nonzero(direct['model']['scene_weight']) > 0
            states.append(direct)
        changed = copy.deepcopy(data); changed['z'].motion[:, 6] += 1e4
        changed['z'].scene[:, 6] += 1e4; changed['truth'][:, 6] += 1e4
        other = base.fit(changed, arm='physical_only', settings=SETTINGS, seed=17,
            identity={'fixture':True}, path=tmp_path/'inactive.gz')
    base.exact(states[0]['model'], other['model'])
    for state in states[1:]:
        for key in ('initial_model', 'draw_hash', 'torch_rng', 'sampler_rng', 'parameters'):
            base.exact(states[0][key], state[key])
    assert np.isfinite(api.predict(states[1], x, scene, envelope, np.ones(len(x)))).all()
    with pytest.raises(TypeError):
        api.predict(states[1], x, scene, envelope, np.ones(len(x)), future_endpoint=np.zeros((len(x), 2)))
