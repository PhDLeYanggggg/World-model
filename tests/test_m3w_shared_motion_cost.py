import numpy as np
import pytest
import torch

from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model import m3w_shared_motion_cost as api
from src.world_model.m3w_european_source_forecast import baseline_numpy


def packet():
    t = np.arange(8.)
    h = np.stack((np.column_stack((t, 2*t)), np.column_stack((3*t+4, t-2)),
        np.column_stack((t-7, -t))))
    g, ids, context = encode_scene(h, np.ones((3, 8), bool), np.arange(3))
    p = np.stack([baseline_numpy(h, k)-h[:, -1, None] for k in range(6)], 1)
    p = np.concatenate((p, (p[:, 1]+1)[:, None]), axis=1)
    return dict(geometry=g, shared_context=context, rollouts=p, floor_index=1,
        recordings=np.zeros(3, int), frames=np.full(3, 7), ids=ids,
        sites=np.array(['group0']*3))


def test_causal_schema_and_capacity_matched_input_mask():
    causal = api.features(**packet())
    assert set(causal) == api.base.base.CAUSAL_KEYS
    assert causal['x'].shape == (3, 7, 428)
    z = torch.from_numpy(causal['x'])
    local = api.for_arm(dict(z=z), 'local_history')['z']
    shared = api.for_arm(dict(z=z), 'shared_motion')['z']
    assert local.shape == shared.shape
    torch.testing.assert_close(local[..., :380], shared[..., :380], rtol=0, atol=0)
    assert not torch.count_nonzero(local[..., 380:])
    assert torch.count_nonzero(shared[..., 380:])
    torch.testing.assert_close(z, shared, rtol=0, atol=0)


def test_inference_rejects_label_fields_and_local_arm_ignores_context():
    causal = api.features(**packet())
    pr = dict(mean=np.zeros(428), std=np.ones(428), clip=8., bias=np.zeros(5, np.float32))
    model = api.base.base.CostHead(428, 8, 17, pr['bias'], 'context')
    with torch.no_grad():
        model.action.weight.fill_(.1)
        model.reference.weight.fill_(.1)
    state = dict(identity=dict(feature_arm='local_history'), preprocess=pr,
        settings=dict(width=8), seed=17, model=model.state_dict())
    a = api.predict(state, causal)
    other = dict(causal, x=causal['x'].copy())
    other['x'][..., 380:] += 100
    np.testing.assert_array_equal(a, api.predict(state, other))
    with pytest.raises(ValueError, match='exact causal schema'):
        api.predict(state, dict(causal, future_endpoint=np.ones((3, 2))))
    with pytest.raises(TypeError):
        api.features(**dict(packet(), future_valid=np.ones((3, 12), bool)))


def test_feature_shape_and_unknown_arm_fail_closed():
    p = packet()
    with pytest.raises(ValueError, match='observed-motion48'):
        api.features(**dict(p, shared_context=np.zeros((3, 47), np.float32)))
    with pytest.raises(ValueError, match='feature treatment'):
        api.for_arm(dict(z=torch.zeros(3, 7, 428)), 'oracle')
