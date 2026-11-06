import inspect
import numpy as np
import pytest
from src.data_unification import m3w_visual_context_packet as method


def test_all_actions_use_only_seven_observed_transitions_and_separate_motion():
    h = np.tile(np.c_[np.arange(8)*2+10., np.ones(8)*50], (3, 1, 1))
    paths = np.zeros((3, 7, 12, 2)); paths[:, :, :, 0] = np.arange(7)[None, :, None]*np.arange(1, 13)
    visited = []
    def load(node, f):
        visited.append(f); T = np.eye(3); T[0, 2] = 1
        return dict(supported=True, native_affine=T.tolist(), inlier_fraction=.9, matches=80,
            residual_median=.1, fb_median=.1, spatial_coverage=.8,
            previous_background_fraction=.7, current_background_fraction=.7,
            log_scale=0., rotation_cos=1., rotation_sin=0.)
    result = method.packet(h, paths, np.zeros(3, int), np.ones(3, int)*7, ['node'], load, lambda n, q: [100, 100])
    assert visited == list(range(1, 8))
    assert result['motion'].shape == (3, 7, 56) and not np.array_equal(result['motion'], result['quality'])
    a = method.select(result, 'image_bounds'); b = method.select(result, 'no_image')
    assert not a[..., :48].any() and not b.any()
    np.testing.assert_array_equal(a[..., 48:], result['motion'][..., 48:])
    assert not {'target', 'valid', 'future'}&set(inspect.signature(method.packet).parameters)


def test_exact_id_alignment_no_nearest_row_or_future_labels():
    from scripts.run_m3w_moving_visual_risk import align, OWN_INPUT_KEYS, OLD_INPUT_KEYS
    keys = np.array([30, 10, 20]); values = np.array([[3, 4], [1, 2], [2, 3]])
    np.testing.assert_array_equal(align(np.array([10, 30]), keys, values), [[1, 2], [3, 4]])
    with pytest.raises((AssertionError, ValueError)):
        align(np.array([11]), keys, values)
    assert not {'moments', 'observed', 'target', 'target_eval', 'valid'}&set(OWN_INPUT_KEYS+OLD_INPUT_KEYS)
