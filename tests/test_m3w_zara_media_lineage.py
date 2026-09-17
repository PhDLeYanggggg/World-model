import numpy as np
import pytest

from src.evaluation.m3w_zara_media_lineage import (
    read_vsp, interpolate_controls, image_coordinates, trace_rows, history_support,
)


def test_vsp_counts_comments_and_negative_coordinates(tmp_path):
    path = tmp_path / 'annotation.vsp'
    path.write_text('1 - the number of splines\n2 - Num of control points\n-2 4 0 90 - point\n3 -5 10 80 - point\n')
    tracks = read_vsp(path)
    np.testing.assert_array_equal(tracks[0], [[-2, 4, 0, 90], [3, -5, 10, 80]])
    path.write_text(path.read_text() + '1\n')
    with pytest.raises(ValueError):
        read_vsp(path)


def test_interpolation_records_later_control_and_exact_queries():
    c = np.array([[0, 0, 0, 0], [10, 20, 10, 0], [20, 0, 20, 0]])
    xy, latest, exact = interpolate_controls(c, np.array([0, 5, 10, 15, 20]))
    np.testing.assert_array_equal(xy, [[0, 0], [5, 10], [10, 20], [15, 10], [20, 0]])
    np.testing.assert_array_equal(latest, [0, 10, 10, 20, 20])
    np.testing.assert_array_equal(exact, [True, False, True, False, True])
    with pytest.raises(ValueError):
        interpolate_controls(c, [-1])


def test_centered_vsp_to_image_has_vertical_flip():
    np.testing.assert_array_equal(image_coordinates([[0, 0], [10, -20]], 720, 576), [[360, 288], [370, 308]])


def test_row_offset_is_explicit_no_extrapolation():
    points = np.array([[1, 1, 0, 0], [11, 1, 0, 0], [21, 1, 0, 0]])
    controls = [np.array([[0, 0, 0, 0], [20, 0, 20, 0]])]
    traced = trace_rows(points, controls, 1)
    np.testing.assert_array_equal(traced['centered_xy'], [[0, 0], [10, 0], [20, 0]])
    assert traced['available'].all()
    assert trace_rows(points, controls, 0)['available'].tolist() == [True, True, False]


def test_history_requires_past_continuity_but_not_future_track_survival():
    points = np.column_stack([np.arange(1, 82, 10), np.ones(9), np.zeros((9, 2))])
    controls = [np.array([[0, 0, 0, 0], [100, 0, 100, 0]])]
    traced = trace_rows(points, controls, 1)
    windows, strict = history_support(points, traced['latest_control_frame'], traced['available'], 1)
    assert windows.shape == (2, 8) and not strict.any()
    w2, _ = history_support(points[:-1], traced['latest_control_frame'][:-1], traced['available'][:-1], 1)
    np.testing.assert_array_equal(w2[0], windows[0])
    assert len(w2) == 1


def test_exact_current_control_suffices_for_annotation_construction_boundary():
    points = np.column_stack([np.arange(1, 72, 10), np.ones(8), np.zeros((8, 2))])
    controls = [np.array([[0, 0, 0, 0], [70, 0, 70, 0]])]
    traced = trace_rows(points, controls, 1)
    windows, strict = history_support(points, traced['latest_control_frame'], traced['available'], 1)
    assert windows.shape == (1, 8) and strict.tolist() == [True]
