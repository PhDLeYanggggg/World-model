import copy
import pytest
import torch
from torch import nn

from src.world_model.m3w_frozen_interaction_capture import InteractionCapture
from src.world_model.m3w_eqmotion_adapter import INPUT_KEYS


class Layer(nn.Module):
    def forward(self, x): return x+1,x,None


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();self.core.gcl_2=Layer()

    def forward(self, inputs):
        h,_,_=self.core.gcl_2(inputs['history'])
        return h[:,0,:24].reshape(len(h),12,2)


def test_passive_hook_preserves_predictions_state_rng_and_removes_itself():
    model=Tiny().eval();x={k:torch.ones(4,3,64) for k in INPUT_KEYS}
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state();plain=model(x)
    with InteractionCapture(model) as capture:
        a,h=capture.run(x);b,j=capture.run(x)
        torch.testing.assert_close(a,plain,rtol=0,atol=0);torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(h,j,rtol=0,atol=0);assert h.shape==(4,64)
        with pytest.raises(ValueError):capture.run(dict(x,future_endpoint=torch.zeros(4,2)))
    assert not model.core.gcl_2._forward_hooks and not model.training
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_layer_identity_is_explicit_not_guessed_after_outcomes():
    with pytest.raises(ValueError):InteractionCapture(Tiny(),suffix='core.missing')
    with pytest.raises(ValueError):InteractionCapture(Tiny(),dimension=32)
