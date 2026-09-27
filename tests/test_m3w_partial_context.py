import numpy as np
import pytest
import torch
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_european_source_forecast import SourceForecaster, pack_scene, baseline_torch
from src.world_model.m3w_observation_quality import masked_neighbors, repair_geometry
from src.world_model.m3w_partial_context import condition_partial_inputs, PartialContextSourceForecaster
from src.world_model.m3w_context_conditioning import condition_inputs


def inputs(partial=True):
    h = np.zeros((2, 8, 2)); h[0, :, 0] = np.arange(8)
    h[1, :, 0] = np.arange(8)+20
    valid = np.ones((2, 8), bool)
    if partial: valid[1, :6] = False
    h[~valid] = 0
    s = dict(history_xy=h, history_valid=valid, target_eligible=valid.all(1),
             agent_id=np.arange(2), baseline_cv=np.zeros((2, 12, 2)))
    g, _ = pack_scene(s)
    return pack_geometry(repair_geometry(g, masked_neighbors(s)))


def test_complete_context_matches_previous_scale():
    x = inputs(False)
    a, sa = condition_inputs(x); b, sb = condition_partial_inputs(x)
    torch.testing.assert_close(sa, sb, rtol=0, atol=0)
    for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_partial_observations_affect_scale_but_masked_garbage_does_not():
    x = inputs(); _, old = condition_inputs(x); a, scale = condition_partial_inputs(x)
    assert (scale > old).all()
    x['neighbors'][~x['neighbor_mask']] = float('nan')
    b, sb = condition_partial_inputs(x)
    torch.testing.assert_close(scale, sb, rtol=0, atol=0)
    torch.testing.assert_close(a['neighbors'], b['neighbors'], rtol=0, atol=0)
    assert torch.isfinite(b['neighbors']).all()


def test_alignment_and_future_fields_rejected():
    x = inputs()
    with pytest.raises(ValueError): condition_partial_inputs({**x, 'future_endpoint': torch.ones(1, 2)})
    x['neighbors'][0, 0, 6, 2] = .1
    with pytest.raises(ValueError): condition_partial_inputs(x)
    x['neighbors'][0, 0, 6, 2] = -.4
    with pytest.raises(ValueError): condition_partial_inputs(x)


def test_same_parameter_budget_initial_floor_and_finite_gradient():
    torch.manual_seed(17)
    old = SourceForecaster(dict(width=8, heads=2, layers=1, neighbor_policy='complete_aligned_history',
        input_conditioning='observed_joint_max_norm', output_parameterization='motion_bounded'), 1)
    torch.manual_seed(17)
    new = PartialContextSourceForecaster(1, width=8, heads=2, layers=1)
    assert old.state_dict().keys() == new.state_dict().keys()
    for k,v in old.state_dict().items(): torch.testing.assert_close(v, new.state_dict()[k], rtol=0, atol=0)
    x = inputs(); y = new(x)
    torch.testing.assert_close(y, baseline_torch(x['history'], 1), rtol=0, atol=0)
    ((y-1)**2).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())


def test_attention_uses_partial_tokens_after_nonzero_output_readout():
    torch.manual_seed(29)
    model = PartialContextSourceForecaster(1, width=8, heads=2, layers=1).eval()
    core = model.predictor.predictor.predictor
    torch.nn.init.normal_(core.output[-1].weight, std=.02)
    x = inputs(); a = model(x)
    changed = {k:v.clone() for k,v in x.items()}
    changed['neighbors'][0, 0, 6:, 1] += 3
    b = model(changed)
    assert not torch.equal(a, b)
    changed = {k:v.clone() for k,v in x.items()}
    changed['neighbors'][~changed['neighbor_mask']] = 1e10
    torch.testing.assert_close(a, model(changed), rtol=0, atol=0)
