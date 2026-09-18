import numpy as np
import pytest

from src.world_model.m3w_observed_unit_frame import observed_unit_frame, restore_delta
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter
from src.data_unification.m3w_causal_recordings import causal_coordinate_transform


def source(kind, factor=1., shift=(0., 0.), change_future=False):
    rows = []
    for agent in range(1 if kind == 'isolated_static' else 2):
        for frame in range(20):
            if agent == 0:
                xy = np.array([0., 0.]) if kind != 'tiny_motion' else np.array([frame*1e-7, frame*2e-7])
            else:
                xy = np.array([3.+frame*.1, 2.])
            if change_future and frame > 7:
                xy += [500., -400.]
            xy = factor*xy+shift
            half = factor*np.array([.2, .4])
            rows.append([agent, *(xy-half), *(xy+half), frame, 0, 0, 0])
    rows = np.asarray(rows)
    return SDDStepAdapter(rows, np.full(len(rows), 'Pedestrian'), 'synthetic/video0', 1)


@pytest.mark.parametrize('kind', ['static_with_neighbor', 'tiny_motion', 'isolated_static'])
def test_coordinate_unit_invariance_and_prediction_restoration(kind):
    base = source(kind)
    x, scale, rotation, supported = observed_unit_frame(base.get_geometry(0)[None])
    delta = np.broadcast_to([.2, -.1], (1, 12, 2)).copy()
    old = base.points[:8]
    t = causal_coordinate_transform(old[:, 2:], old[:, 0], 12)
    canonical_native = restore_delta(delta, scale, rotation, supported)[0]*t['scale']@t['rotation'].T
    for factor in (1e-4, 1e4):
        other = source(kind, factor)
        z, s, q, ok = observed_unit_frame(other.get_geometry(0)[None])
        np.testing.assert_allclose(z, x, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(ok, supported)
        h = other.points[:8]
        tt = causal_coordinate_transform(h[:, 2:], h[:, 0], 12)
        native = restore_delta(delta, s, q, ok)[0]*tt['scale']@tt['rotation'].T
        np.testing.assert_allclose(native/factor, canonical_native, atol=3e-6, rtol=3e-6)


def test_future_labels_and_translation_cannot_change_frame():
    before = observed_unit_frame(source('static_with_neighbor').get_geometry(0)[None])
    after = observed_unit_frame(source('static_with_neighbor', shift=(90., -60.), change_future=True).get_geometry(0)[None])
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)


def test_no_observed_spatial_anchor_means_no_invented_scale_or_correction():
    x, scale, q, ok = observed_unit_frame(source('isolated_static').get_geometry(0)[None])
    assert not ok.any() and not scale.any()
    np.testing.assert_array_equal(restore_delta(np.ones((1, 12, 2)), scale, q, ok), 0.)
    assert np.isfinite(x).all()


def test_future_time_or_invalid_mask_fails_closed():
    x = source('static_with_neighbor').get_geometry(0)[None]
    bad = x.copy(); bad[0, 16] = .1
    with pytest.raises(ValueError, match='past'):
        observed_unit_frame(bad)
    bad = x.copy(); bad[0, 230] = .3
    with pytest.raises(ValueError, match='mask'):
        observed_unit_frame(bad)


def test_raw_dimensional_summary_fields_are_not_invariant_input_shortcuts():
    x = source('static_with_neighbor').get_geometry(0)[None]
    y = x.copy()
    y[:, [294, 295, 296, 297, 299, 301, 302, 303, 304, 305, 306, 307]] *= 1000
    for a, b in zip(observed_unit_frame(x), observed_unit_frame(y)):
        np.testing.assert_array_equal(a, b)
