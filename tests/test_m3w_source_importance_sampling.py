import itertools

import numpy as np
import pytest
import torch

from src.world_model.m3w_source_importance_sampling import (
    importance_objective, uniform_risk_factors, fit_importance_candidate,
)
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate
from src.world_model.m3w_source_temporal_centered import CenteredTemporalDynamics


def test_exact_expected_batch_loss_and_gradient_not_self_normalized():
    p = np.array([.1, .2, .7]); factors = torch.from_numpy(uniform_risk_factors(p))
    torch.manual_seed(9)
    x = torch.randn(3, 12, 2, dtype=torch.float64)
    target = torch.randn(3, 12, 2, dtype=torch.float64); target[0] = 0
    theta = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    expected, uncorrected, self_normalized = [], [], []
    for draw in itertools.product(range(3), repeat=2):
        ids = list(draw); probability = np.prod(p[ids])
        loss, plain, _ = importance_objective(theta*x[ids], target[ids], 2., factors[ids])
        expected.append(probability*loss); uncorrected.append(probability*plain)
        self_normalized.append(probability*loss/factors[ids].mean())
    reference = importance_objective(theta*x, target, 2., torch.ones(3, dtype=torch.float64))[0]
    exact = torch.stack(expected).sum()
    torch.testing.assert_close(exact, reference, atol=1e-12, rtol=0)
    grad = lambda value:torch.autograd.grad(value, theta, retain_graph=True)[0]
    torch.testing.assert_close(grad(exact), grad(reference), atol=1e-12, rtol=0)
    assert abs(float(torch.stack(uncorrected).sum().detach()-reference.detach())) > .01
    assert abs(float(torch.stack(self_normalized).sum().detach()-reference.detach())) > .001


@pytest.mark.parametrize('p', [[], [0.,1.], [-.1,1.1], [.2,.2], [np.nan,1.], [[1.]]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError): uniform_risk_factors(p)


def test_factors_use_training_members_only():
    ids = np.arange(6); groups = np.array(['a','a','a','b','b','c'])
    p,_ = episode_weights(ids[:4], ids, groups)
    np.testing.assert_allclose(uniform_risk_factors(p), [1.5,1.5,1.5,.5])
    groups[4:] = 'a'
    np.testing.assert_array_equal(uniform_risk_factors(p),
                                 uniform_risk_factors(episode_weights(ids[:4],ids,groups)[0]))


def test_resume_exact_and_changed_sampler_rejected(tmp_path):
    torch.set_num_threads(4); torch.manual_seed(4)
    x = (torch.randn(4,480), torch.randn(4,8,512), torch.ones(4,8))
    y = torch.randn(4,12,2)*.1; y[0] = 0
    ids = np.arange(4); p,_ = episode_weights(ids, ids, np.array(['a','a','a','b']))
    config = dict(start_step=2,updates=6,batch_size=2,learning_rate=.0003,
                  minimum_lr_ratio=.01,weight_decay=.0001,checkpoint_every=2)
    def inputs(q):
        return tuple(v[q] for v in x), (torch.ones(len(q)),torch.eye(2).expand(len(q),2,2),torch.ones(len(q),dtype=torch.bool))
    def fit(path, stop=None, probs=p, old=False):
        torch.manual_seed(17); model = CenteredTemporalDynamics('centered')
        extra = dict(suppress_zero_targets=False) if old else {}
        fn = fit_motion_candidate if old else fit_importance_candidate
        result = fn(model,inputs,lambda q:y[q],ids,probs,scale=1.,seed=17,config=config,
            identity={'fixed':'fixture'},checkpoint=path,heartbeat=lambda **kw:None,stop_at=stop,**extra)
        return model,result
    full,_ = fit(tmp_path/'full.pt'); fit(tmp_path/'resume.pt',2)
    resumed,result = fit(tmp_path/'resume.pt')
    for k,v in full.state_dict().items(): torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
    a = torch.load(tmp_path/'full.pt',weights_only=False); b = torch.load(tmp_path/'resume.pt',weights_only=False)
    np.testing.assert_array_equal(a['draw_counts'],b['draw_counts'])
    np.testing.assert_array_equal(a['sampler_rng'],b['sampler_rng'])
    assert result['new_updates'] == 4 and fit(tmp_path/'resume.pt')[1]['new_updates'] == 0
    with pytest.raises(AssertionError): fit(tmp_path/'resume.pt',probs=np.full(4,.25))
    corrected,_ = fit(tmp_path/'uniform.pt',probs=np.full(4,.25))
    original,_ = fit(tmp_path/'original.pt',probs=np.full(4,.25),old=True)
    for k,v in corrected.state_dict().items(): torch.testing.assert_close(v,original.state_dict()[k],atol=0,rtol=0)
