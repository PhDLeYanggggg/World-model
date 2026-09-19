import numpy as np
import torch

from src.world_model.m3w_source_conditioned_readout import ConditionedTemporalDynamics
from src.world_model.m3w_source_importance_sampling import fit_importance_candidate


def test_conditioned_readout_resume_is_exact_including_buffer_and_optimizer(tmp_path):
    torch.manual_seed(101)
    geometry = torch.randn(7, 480)
    appearance = torch.randn(7, 8, 512)
    coverage = torch.ones(7, 8)
    target = torch.randn(7, 12, 2)*.01
    ids = np.arange(7)
    probabilities = np.arange(1, 8, dtype=float)/28
    config = dict(start_step=2, updates=5, batch_size=4, learning_rate=.0003,
                  minimum_lr_ratio=.01, weight_decay=.0001, checkpoint_every=2)
    identity = dict(test='conditioned_resume', readout_gain=540.)

    def inputs(q):
        return ((geometry[q], appearance[q], coverage[q]),
                (torch.ones(len(q)), torch.eye(2).expand(len(q), -1, -1), torch.ones(len(q), dtype=torch.bool)))

    def fit(model, path, stop=None):
        return fit_importance_candidate(model, inputs, lambda q: target[q], ids, probabilities,
            scale=1., seed=17, config=config, identity=identity, checkpoint=path,
            heartbeat=lambda **kw: None, stop_at=stop)

    torch.manual_seed(17)
    whole = ConditionedTemporalDynamics('centered', 540.)
    fit(whole, tmp_path/'whole.pt')
    torch.manual_seed(17)
    interrupted = ConditionedTemporalDynamics('centered', 540.)
    assert not fit(interrupted, tmp_path/'resume.pt', 2)['complete']
    recovered = ConditionedTemporalDynamics('centered', 540.)
    assert fit(recovered, tmp_path/'resume.pt')['new_updates'] == 3
    for name, value in whole.state_dict().items():
        assert torch.equal(value, recovered.state_dict()[name])
    a = torch.load(tmp_path/'whole.pt', weights_only=False)
    b = torch.load(tmp_path/'resume.pt', weights_only=False)
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])
    np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
    assert a['trace'] == b['trace']
    assert fit(recovered, tmp_path/'resume.pt')['new_updates'] == 0
