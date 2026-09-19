import numpy as np
import pytest
import torch

from src.evaluation.m3w_source_gradient_diagnostic import (
    flat_gradient, gradient_relation, clipped_gradient, layer_energy, output_shrink_curve,
)


def test_partitioned_gradient_matches_full_objective_with_unused_parameter():
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    unused = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    params = list(model.parameters()) + [unused]
    x = torch.tensor([[1., 0.], [0., 1.], [2., -1.]], dtype=torch.float64)
    y = torch.tensor([[0., 0.], [1., 2.], [0., 0.]], dtype=torch.float64)
    loss = torch.linalg.vector_norm(model(x)-y, dim=-1)
    full = flat_gradient(loss.mean(), params, retain_graph=True)
    a = flat_gradient(loss[[0, 2]].sum()/3, params, retain_graph=True)
    b = flat_gradient(loss[[1]].sum()/3, params)
    np.testing.assert_allclose(full, a+b, atol=1e-15, rtol=1e-15)
    assert not full[-3:].any()


def test_clip_before_mean_changes_direction_not_just_magnitude():
    gradients = np.array([[100., 0.], [0., 1.]])
    mean = gradients.mean(0)
    after = np.mean([clipped_gradient(g, 5) for g in gradients], axis=0)
    assert gradient_relation(after, mean)['cosine'] < .99
    assert gradient_relation(clipped_gradient(mean, 5), mean)['cosine'] == pytest.approx(1.)


def test_zero_gradient_is_undefined_direction_not_perfect_alignment():
    assert gradient_relation(np.zeros(2), np.zeros(2))['cosine'] is None
    np.testing.assert_array_equal(clipped_gradient(np.zeros(2), 5), np.zeros(2))
    with pytest.raises(ValueError):
        clipped_gradient(np.ones(2), 0)


def test_layer_accounting_and_shape_guard():
    params = [('geometry.weight', torch.nn.Parameter(torch.ones(2))),
              ('head.2.bias', torch.nn.Parameter(torch.ones(1)))]
    result = layer_energy(np.array([3., 4., 5.]), params)
    assert result['geometry']['squared_norm_share'] == .5
    assert result['output_head']['norm'] == 5
    with pytest.raises(ValueError):
        layer_energy(np.zeros(4), params)


def test_shrink_curve_keeps_static_percentage_undefined_and_no_selection():
    target = np.zeros((2, 12, 2)); target[1, :, 0] = 1
    pred = np.ones_like(target)*.1
    result = output_shrink_curve(pred, target, np.ones(2), 2., [0, .5, 1])
    assert result[0]['gain_percent'] == 0
    assert result[0]['static_pixel_harm'] == 0
    assert result[-1]['static_pixel_harm'] > 0
    assert len(result) == 3
    with pytest.raises(ValueError):
        output_shrink_curve(pred, target, np.ones(2), 2., [1.1])
