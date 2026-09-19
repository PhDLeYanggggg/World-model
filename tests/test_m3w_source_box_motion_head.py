import numpy as np
import pytest
import torch

from src.world_model.m3w_source_box_motion_head import BoxMotionDynamics
from src.world_model.m3w_source_importance_sampling import fit_importance_candidate


def test_same_parameter_count_zero_start_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    geometry = torch.randn(7, 480); tokens = torch.zeros(7, 8, 512)
    tokens[:, 1:, :19] = torch.randn(7, 7, 19)
    coverage = torch.ones(7, 8); target = torch.randn(7, 12, 2)*.01
    ids = np.arange(7); probabilities = np.arange(1, 8)/28.
    config = dict(start_step=2, updates=5, batch_size=4, learning_rate=.0003,
                  minimum_lr_ratio=.01, weight_decay=.0001, checkpoint_every=2)
    def inputs(q):
        return ((geometry[q], tokens[q], coverage[q]), (torch.ones(len(q)),
            torch.eye(2).expand(len(q), -1, -1), torch.ones(len(q), dtype=torch.bool)))
    def fit(model, path, stop=None):
        return fit_importance_candidate(model, inputs, lambda q: target[q], ids, probabilities,
            scale=1., seed=17, config=config, identity={'test':'box_motion_resume'}, checkpoint=path,
            heartbeat=lambda **kw: None, stop_at=stop)
    torch.manual_seed(17); whole = BoxMotionDynamics(540.)
    assert sum(v.numel() for v in whole.parameters()) == 63960
    assert not whole(geometry, tokens, coverage).any()
    fit(whole, tmp_path/'whole.pt')
    torch.manual_seed(17); interrupted = BoxMotionDynamics(540.)
    assert not fit(interrupted, tmp_path/'resume.pt', 2)['complete']
    recovered = BoxMotionDynamics(540.)
    assert fit(recovered, tmp_path/'resume.pt')['new_updates'] == 3
    for k, value in whole.state_dict().items(): assert torch.equal(value, recovered.state_dict()[k])
    a = torch.load(tmp_path/'whole.pt', weights_only=False); b = torch.load(tmp_path/'resume.pt', weights_only=False)
    assert a['trace'] == b['trace'] and torch.equal(a['sampler_rng'], b['sampler_rng'])
    np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
    assert fit(recovered, tmp_path/'resume.pt')['new_updates'] == 0


@pytest.mark.parametrize('gain', [0, -1, float('nan'), float('inf')])
def test_invalid_gain_rejected(gain):
    with pytest.raises(ValueError): BoxMotionDynamics(gain)
