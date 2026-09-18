import copy
import hashlib

import numpy as np
import pytest
import torch

from src.world_model.m3w_source_continuation import continue_fit
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, fit_dynamics, forecast
from src.world_model.m3w_source_modality_continuation import continue_modality


def setup(tmp_path, arm):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    x = (torch.randn(8, 6), torch.rand(8, 8, 3, 32, 32), torch.ones(8, 8, 1, 32, 32))
    frame = (torch.ones(8), torch.eye(2)[None].repeat(8, 1, 1), torch.ones(8, dtype=torch.bool))
    target = torch.randn(8, 12, 2)*.1
    ids, weights = np.arange(8), np.ones(8)/8
    def inputs(index):
        return tuple(a[index] for a in x), tuple(a[index] for a in frame)
    parent_cfg = dict(updates=3, batch_size=4, learning_rate=.0003, weight_decay=.0001, checkpoint_every=2)
    torch.manual_seed(17)
    parent_model = SourceDynamics(6)
    fit_dynamics(parent_model, inputs, lambda i:target[i], ids, weights, arm=arm, objective='ade', normalizer=1.,
        seed=17, config=parent_cfg, identity={'arm':arm}, checkpoint=tmp_path/'parent.pt', heartbeat=lambda **_:None)
    parent = torch.load(tmp_path/'parent.pt', weights_only=False)
    config = dict(parent_cfg, start_step=3, updates=10, minimum_lr_ratio=.01, milestones=[3, 6, 10])
    return inputs, target, ids, weights, parent, config, parent_cfg


def run(tmp_path, name, payload, arm, schedule='cosine', stop=None, engine=continue_modality, inputs_override=None):
    inputs, target, ids, weights, parent, config, _ = payload
    model = SourceDynamics(6)
    kw = {} if engine is continue_fit else {'arm':arm}
    result = engine(model, inputs_override or inputs, lambda i:target[i], ids, weights,
        parent=parent, schedule=schedule, config=config, identity={'arm':arm, 'schedule':schedule},
        checkpoint=tmp_path/name, heartbeat=lambda **_:None, snapshot=lambda _:None, stop_at=stop, **kw)
    return model, result


@pytest.mark.parametrize('schedule', ['constant', 'cosine'])
def test_rgb_engine_exactly_matches_frozen_predecessor(tmp_path, schedule):
    payload = setup(tmp_path, 'past_rgb')
    a, _ = run(tmp_path, 'old.pt', payload, 'past_rgb', schedule, engine=continue_fit)
    b, _ = run(tmp_path, 'new.pt', payload, 'past_rgb', schedule)
    np.testing.assert_array_equal(forecast(a, payload[0], payload[2], 'past_rgb'),
                                  forecast(b, payload[0], payload[2], 'past_rgb'))


def test_mask_constant_matches_original_uninterrupted_fit(tmp_path):
    payload = setup(tmp_path, 'mask_only')
    a, _ = run(tmp_path, 'continued.pt', payload, 'mask_only', 'constant')
    inputs, target, ids, weights, _, _, config = payload
    torch.manual_seed(17)
    b = SourceDynamics(6)
    fit_dynamics(b, inputs, lambda i:target[i], ids, weights, arm='mask_only', objective='ade', normalizer=1.,
        seed=17, config=dict(config, updates=10), identity={'full':True}, checkpoint=tmp_path/'full.pt', heartbeat=lambda **_:None)
    np.testing.assert_array_equal(forecast(a, inputs, ids, 'mask_only'), forecast(b, inputs, ids, 'mask_only'))


def test_mask_training_is_invariant_to_rgb_and_preserves_parent(tmp_path):
    payload = setup(tmp_path, 'mask_only')
    original = copy.deepcopy(payload[4])
    def changed(i):
        features, frame = payload[0](i)
        return (features[0], features[1]*500+100, features[2]), frame
    a, _ = run(tmp_path, 'a.pt', payload, 'mask_only')
    b, _ = run(tmp_path, 'b.pt', payload, 'mask_only', inputs_override=changed)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])
    for k, v in original['optimizer']['state'].items():
        for name in v:
            assert torch.equal(v[name], payload[4]['optimizer']['state'][k][name])


def test_mask_resume_exact_and_completed_readonly(tmp_path):
    payload = setup(tmp_path, 'mask_only')
    a, _ = run(tmp_path, 'a.pt', payload, 'mask_only')
    run(tmp_path, 'b.pt', payload, 'mask_only', stop=7)
    b, result = run(tmp_path, 'b.pt', payload, 'mask_only')
    assert result['new_updates'] == 3
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])
    digest = hashlib.sha256((tmp_path/'b.pt').read_bytes()).hexdigest()
    assert run(tmp_path, 'b.pt', payload, 'mask_only')[1]['new_updates'] == 0
    assert digest == hashlib.sha256((tmp_path/'b.pt').read_bytes()).hexdigest()


def test_parent_modality_cannot_be_relabelled(tmp_path):
    payload = setup(tmp_path, 'mask_only')
    with pytest.raises(ValueError, match='parent'):
        run(tmp_path, 'wrong.pt', payload, 'past_rgb')
    with pytest.raises(ValueError, match='Fixed'):
        run(tmp_path, 'wrong.pt', payload, 'unknown')
