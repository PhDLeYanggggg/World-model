import copy
import numpy as np
import torch
from src.world_model import m3w_exact_region_cost as api
from src.world_model import m3w_policy_region_cost as old
from src.evaluation.m3w_region_gradient_diagnosis import sampling
from tests.test_m3w_scene_context_cost import scene_sample, SETTINGS
from tests.test_m3w_policy_region_cost import regions_for


def test_chunked_region_gradient_matches_direct_autograd():
    torch.set_num_threads(1); *_, data = scene_sample()
    regions = regions_for(data, 'physical_neural')
    # Two different finite regions,including a zero target component.
    active = torch.nonzero(regions['pool'] >= 0).flatten()
    regions['pool'][active[::2]] = 1
    regions['target'] = torch.cat((regions['target'], regions['target']*.5))
    regions['target'][0,4] = 0.
    model = old.SceneCostHead(data['z'].shape[-1], SETTINGS['width'], 17, data['preprocess']['bias'])
    with torch.no_grad():
        model.network[-1].weight.normal_(0,.02)
    actions = torch.arange(7); rms = torch.from_numpy(data['preprocess']['rms'])
    p = api.prediction(model, data, actions, active.numpy())
    direct = old.region_loss(p.double(), active.numpy(), regions, rms.double())
    direct.backward(); expected = api.gradient_vector(model)
    result = api.exact_region_pass(model, data, regions, actions, rms, chunk_size=3, backward=True)
    np.testing.assert_allclose(result['gradient'], expected, rtol=1e-5, atol=1e-7)
    assert abs(result['loss']-float(direct.detach())) < 1e-6
    before = copy.deepcopy(model.state_dict())
    api.exact_region_pass(model, data, regions, actions, rms, chunk_size=2, pool_id=1, backward=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_original_sampler_hash_and_unknown_mask_are_replayed():
    *_, data = scene_sample(); regions = regions_for(data, 'physical_neural')
    result = sampling(data, regions, SETTINGS, 17, probes=2)
    assert len(result['batches']) == 2
    assert sum(result['row_draws_per_region']) <= SETTINGS['steps']*SETTINGS['batch_size']
    other = sampling(data, regions, SETTINGS, 17, probes=2)
    assert result['draw_hash'] == other['draw_hash']
    assert result['visits'] == other['visits']
