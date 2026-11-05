import numpy as np
import pytest
import torch
from src.evaluation import m3w_neighbor_order as method
from tests.test_m3w_harm_floor_risk import prepared


def test_permutation_moves_complete_history_with_masks_and_is_invertible():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 7, 428)).astype('float32')
    for k in range(8):
        p = method.permute_neighbors(x, k)
        np.testing.assert_array_equal(method.permute_neighbors(p, (-k) % 8), x)
        np.testing.assert_array_equal(p[..., :32], x[..., :32])
        np.testing.assert_array_equal(p[..., 288:], x[..., 288:])
        np.testing.assert_array_equal(p[..., 32:224].reshape(3, 7, 8, 8, 3)[..., k, :, :],
            x[..., 32:224].reshape(3, 7, 8, 8, 3)[..., 0, :, :])
        np.testing.assert_array_equal(p[..., 224:288].reshape(3, 7, 8, 8)[..., k, :],
            x[..., 224:288].reshape(3, 7, 8, 8)[..., 0, :])
    with pytest.raises(ValueError):
        method.permute_neighbors(x, 8)


def test_measurements_match_existing_joint_attribution(tmp_path):
    torch.set_num_threads(1)
    data, origin, settings = prepared(tmp_path)
    # The shared tiny training fixture stores unclipped rollout columns;
    # this audit checks the real inference preprocessing contract as well.
    z = method.features(data['causal']['x'], origin)
    data['z'], data['extra'] = z[..., :428], z[..., 428:]
    origin = dict(origin, input_hash=method.api.parent.kin.digest(z.numpy()))
    state = method.api.fit(data, origin, arm='joint', settings=settings,
        identity={'job': 'fixture'}, path=tmp_path/'fit.pt.gz')
    result = method.audit(state, data)
    full = method.api.full(method.api.model_from(state), data, state['sampler']['counts'])
    joint = method.api.attribution(state, data)['total']
    np.testing.assert_allclose(result['shifts'][0]['mean_MSE'], full['mean_MSE'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result['shifts'][0]['joint_cost'], joint['total_joint_cost'], rtol=1e-6, atol=1e-8)
    assert result['shifts'][0]['false_safe'] == joint['false_safe_count']
    assert result['shifts'][0]['false_reject'] == joint['false_reject_count']


def test_metric_preserves_both_channel_mass_zero_targets_and_unknown_masks():
    y = torch.tensor([[[1., 2.]], [[-1., -2.]], [[0., 0.]], [[100., 100.]]]).expand(-1, 7, -1)
    r = np.zeros((4, 7, 2), np.float32); d = r.copy(); d[1:3] = 1
    legal = np.zeros((4, 7), bool); legal[:, 0] = True
    observed = np.array([True, True, True, False]); weights = np.ones(4)/4
    result = method.measured(r, d, y, legal, observed, weights)
    assert result['joint_cost'] == 1.5
    assert result['false_safe'] == 1
    assert result['false_reject'] == 2
    assert result['admitted'] == 1
