import numpy as np
import pytest

from src.evaluation.m3w_zara_media_lineage import trace_rows
from src.evaluation.m3w_zara_past_media import (
    native_to_image_xy, project_image_xy, select_past_media_controls,
    source_origin_anchor, validate_past_request,
)


def test_translation_anchor_and_inverse_do_not_fit_other_rows():
    h = np.array([[.2, .01, 2], [.03, .4, -1], [.0001, .0002, 1]])
    xy = np.array([[100., 300], [200, 150], [400, 500]])
    projected = project_image_xy(xy, h)
    origin = np.array([-1.2, -15.6])
    native = projected + origin
    anchor = source_origin_anchor(native[0], xy[0], h)
    np.testing.assert_allclose(anchor, origin)
    np.testing.assert_allclose(native_to_image_xy(native, h, anchor), xy)
    # A different held-out coordinate cannot change the one-anchor adapter.
    native[-1] += 100
    np.testing.assert_array_equal(source_origin_anchor(native[0], xy[0], h), anchor)


def test_no_future_survival_filter_for_eight_past_frames():
    points = np.column_stack([np.arange(1, 72, 10), np.ones(8), np.arange(8), np.zeros(8)])
    tracks = [np.array([[0., 0., 0., 0.], [10., 0., 100., 0.]])]
    traced = trace_rows(points, tracks, 1)
    controls = select_past_media_controls(points, traced, 720, 576)
    assert len(controls) == 1
    assert controls[0]['history_source_frames'] == list(range(0, 71, 10))
    assert not controls[0]['source_controls_as_of_query']
    assert validate_past_request(controls[0])
    with pytest.raises(ValueError, match='after the query'):
        validate_past_request(controls[0], require_control_as_of_query=True)


def test_future_frame_rejected_and_source_provenance_not_erased():
    control = {'history_source_frames': list(range(0, 71, 10)),
        'history_image_xy': np.zeros((8, 2)).tolist(),
        'latest_contributing_control_frames': [0] + [70] * 7, 'query_source_frame': 70}
    assert validate_past_request(control, require_control_as_of_query=True)
    control['history_source_frames'][-1] = 80
    with pytest.raises(ValueError, match='past frames'):
        validate_past_request(control)


def test_bad_coordinate_inputs_rejected():
    with pytest.raises(ValueError):
        project_image_xy(np.zeros((2, 3)), np.eye(3))
    with pytest.raises(ValueError):
        source_origin_anchor([0, float('nan')], [0, 0], np.eye(3))
