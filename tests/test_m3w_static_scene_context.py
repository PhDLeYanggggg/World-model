import numpy as np
import pytest

from src.world_model.m3w_static_scene_context import scene_features, signed_distance
from src.world_model.m3w_european_source_forecast import pack_scene


def test_static_polygon_distance_and_scene_features_use_no_future():
    polygon = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    np.testing.assert_allclose(signed_distance(np.array([[5., 5.], [11., 5.], [0., 4.]]), polygon), [5., -1., 0.])
    h = np.tile([5., 5.], (2, 8, 1)); r = np.tile([5., 5.], (2, 12, 1))
    x = scene_features(history=h, polygon=polygon, rollout=r)
    np.testing.assert_array_equal(x[0], x[1])
    shifted = scene_features(history=h+np.array([4., 0.]), polygon=polygon, rollout=r+np.array([4., 0.]))
    assert not np.array_equal(x, shifted)
    with pytest.raises(TypeError):
        scene_features(history=h, polygon=polygon, rollout=r, future_endpoint=np.zeros((2, 2)))


def test_existing_relative_pack_cannot_distinguish_global_image_translation():
    h = np.arange(32).reshape(2, 8, 2).astype(float)
    inputs = dict(history_xy=h, history_valid=np.ones((2, 8), bool), agent_id=np.array([1, 2]),
                  target_eligible=np.ones(2, bool), baseline_cv=np.tile(h[:, -1, None], (1, 12, 1)))
    a, ids = pack_scene(inputs)
    translated = {**inputs, 'history_xy':h+100, 'baseline_cv':inputs['baseline_cv']+100}
    b, other = pack_scene(translated)
    np.testing.assert_array_equal(ids, other)
    np.testing.assert_array_equal(a, b)
