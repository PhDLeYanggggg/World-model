import numpy as np
import pytest
import torch

from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate
from src.world_model.m3w_source_temporal_centered import CenteredTemporalDynamics


def test_equal_mass_per_training_episode_no_held_frequency():
    ids = np.arange(6)
    groups = np.array(['a', 'a', 'a', 'b', 'b', 'c'])
    w, summary = episode_weights(ids[:4], ids, groups)
    np.testing.assert_allclose(w, [1/6, 1/6, 1/6, 1/2])
    assert summary['training_episodes'] == 2
    groups[4:] = 'a'
    np.testing.assert_array_equal(w, episode_weights(ids[:4], ids, groups)[0])
    with pytest.raises(ValueError): episode_weights([7], ids, groups)
    with pytest.raises(ValueError): episode_weights([1, 1], ids, groups)


def test_weighted_resume_has_identical_parameters_and_draws(tmp_path):
    torch.set_num_threads(4); torch.manual_seed(4)
    x = (torch.randn(4,480), torch.randn(4,8,512), torch.ones(4,8))
    y = torch.randn(4,12,2)*.1
    ids = np.arange(4); w,_ = episode_weights(ids, ids, np.array(['a','a','a','b']))
    config = dict(start_step=2,updates=6,batch_size=2,learning_rate=.0003,
                  minimum_lr_ratio=.01,weight_decay=.0001,checkpoint_every=2)
    def inputs(q):
        return tuple(v[q] for v in x), (torch.ones(len(q)),torch.eye(2).expand(len(q),2,2),torch.ones(len(q),dtype=torch.bool))
    def fit(path, stop=None):
        torch.manual_seed(17); model = CenteredTemporalDynamics('centered')
        result = fit_motion_candidate(model, inputs, lambda q:y[q], ids, w, scale=1., seed=17,
            config=config, identity={'weights':w.tolist()}, checkpoint=path, heartbeat=lambda **kw:None,
            suppress_zero_targets=False, stop_at=stop)
        return model,result
    full,_ = fit(tmp_path/'full.pt'); fit(tmp_path/'resume.pt',2); resumed,result = fit(tmp_path/'resume.pt')
    for k,v in full.state_dict().items(): torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
    a = torch.load(tmp_path/'full.pt',weights_only=False); b = torch.load(tmp_path/'resume.pt',weights_only=False)
    np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
    assert result['new_updates'] == 4 and fit(tmp_path/'resume.pt')[1]['new_updates'] == 0
