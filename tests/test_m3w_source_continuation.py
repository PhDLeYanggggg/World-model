import copy
import hashlib

import numpy as np
import pytest
import torch

from src.world_model.m3w_source_continuation import continue_fit, learning_rate
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, fit_dynamics, forecast


def fixture(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    x = (torch.randn(9, 6), torch.rand(9, 8, 3, 32, 32), torch.ones(9, 8, 1, 32, 32))
    frame = (torch.ones(9), torch.eye(2)[None].repeat(9, 1, 1), torch.ones(9, dtype=torch.bool))
    y = torch.randn(9, 12, 2)*.1
    ids, w = np.arange(9), np.ones(9)/9
    def inputs(i):
        return tuple(v[i] for v in x), tuple(v[i] for v in frame)
    base = dict(updates=3, batch_size=4, learning_rate=.0003, weight_decay=.0001, checkpoint_every=2)
    torch.manual_seed(17)
    model = SourceDynamics(6)
    fit_dynamics(model, inputs, lambda i:y[i], ids, w, arm='past_rgb', objective='ade', normalizer=1.,
        seed=17, config=base, identity={'parent': True}, checkpoint=tmp_path/'parent.pt', heartbeat=lambda **_:None)
    parent = torch.load(tmp_path/'parent.pt', weights_only=False)
    cfg = dict(base, start_step=3, updates=10, minimum_lr_ratio=.01, milestones=[3, 6, 10])
    return inputs, y, ids, w, parent, cfg, base


def test_rate_has_registered_endpoints_and_rejects_invalid_steps():
    cfg = dict(start_step=2000, updates=10000, minimum_lr_ratio=.01, learning_rate=.0003)
    assert learning_rate('constant', 9999, cfg) == .0003
    assert learning_rate('cosine', 2000, cfg) == .0003
    assert learning_rate('cosine', 9999, cfg) == pytest.approx(.000003)
    rates = [learning_rate('cosine', s, cfg) for s in range(2000, 10000)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    for schedule, step in [('other', 2000), ('cosine', 1999), ('cosine', 10000)]:
        with pytest.raises(ValueError):
            learning_rate(schedule, step, cfg)


def run(tmp_path, name, payload, schedule='constant', stop=None, snapshots=None):
    inputs, y, ids, w, parent, cfg, _ = payload
    model = SourceDynamics(6)
    fit = continue_fit(model, inputs, lambda i:y[i], ids, w, parent=parent,
        schedule=schedule, config=cfg, identity={'fork': schedule}, checkpoint=tmp_path/name,
        heartbeat=lambda **_:None, snapshot=lambda s: snapshots.append(s['step']) if snapshots is not None else None,
        stop_at=stop)
    return model, fit


def test_constant_continuation_matches_uninterrupted_parent_algorithm(tmp_path):
    payload = fixture(tmp_path)
    inputs, y, ids, w, _, _, base = payload
    a, result = run(tmp_path, 'continuation.pt', payload)
    torch.manual_seed(17)
    b = SourceDynamics(6)
    fit_dynamics(b, inputs, lambda i:y[i], ids, w, arm='past_rgb', objective='ade', normalizer=1.,
        seed=17, config=dict(base, updates=10), identity={'full':True}, checkpoint=tmp_path/'full.pt', heartbeat=lambda **_:None)
    np.testing.assert_array_equal(forecast(a, inputs, ids, 'past_rgb'), forecast(b, inputs, ids, 'past_rgb'))
    assert result['new_updates'] == result['additional_updates'] == 7


@pytest.mark.parametrize('schedule', ['constant', 'cosine'])
def test_exact_interrupted_resume_and_completed_readonly(tmp_path, schedule):
    payload = fixture(tmp_path)
    snapshots = []
    a, _ = run(tmp_path, 'all.pt', payload, schedule, snapshots=snapshots)
    assert snapshots == [3, 6, 10]
    run(tmp_path, 'resume.pt', payload, schedule, stop=7)
    b, result = run(tmp_path, 'resume.pt', payload, schedule)
    inputs, _, ids, *_ = payload
    np.testing.assert_array_equal(forecast(a, inputs, ids, 'past_rgb'), forecast(b, inputs, ids, 'past_rgb'))
    assert result['new_updates'] == 3
    before = hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()
    assert run(tmp_path, 'resume.pt', payload, schedule)[1]['new_updates'] == 0
    assert before == hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()


def test_schedules_share_stream_and_do_not_mutate_parent(tmp_path):
    payload = fixture(tmp_path)
    original = copy.deepcopy(payload[4])
    run(tmp_path, 'constant.pt', payload)
    run(tmp_path, 'cosine.pt', payload, 'cosine')
    a, b = [torch.load(tmp_path/name, weights_only=False) for name in ('constant.pt', 'cosine.pt')]
    np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    for key, tensor in original['model'].items():
        assert torch.equal(tensor, payload[4]['model'][key])
    for key, values in original['optimizer']['state'].items():
        for name, tensor in values.items():
            assert torch.equal(tensor, payload[4]['optimizer']['state'][key][name])
    assert a['optimizer']['param_groups'][0]['lr'] != b['optimizer']['param_groups'][0]['lr']


def test_resume_rejects_changed_identity_schedule_and_rows(tmp_path):
    payload = fixture(tmp_path)
    run(tmp_path, 'branch.pt', payload, stop=5)
    with pytest.raises(ValueError, match='identity/config/schedule'):
        run(tmp_path, 'branch.pt', payload, 'cosine')
    bad = list(payload)
    bad[2] = payload[2][::-1]
    with pytest.raises(AssertionError):
        run(tmp_path, 'new.pt', bad)
