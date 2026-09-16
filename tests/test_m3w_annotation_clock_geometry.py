import struct

import numpy as np
import pytest

from scripts.audit_m3w_annotation_clock_geometry import avi_header, project_points, align_source_rows, clock_status
from src.evaluation.m3w_recording_lineage import read_track


def test_video_header_without_decoding_and_truncation_refusal(tmp_path):
    fields = [40000, 0, 0, 0, 100, 0, 1, 0, 640, 480]
    chunk = b'avih' + struct.pack('<I', 40) + struct.pack('<10I', *fields)
    content = b'AVI LIST' + struct.pack('<I', len(chunk)+4) + b'hdrl' + chunk
    path = tmp_path / 'video.avi'
    path.write_bytes(b'RIFF'+struct.pack('<I', len(content))+content)
    assert avi_header(path) == {'microseconds_per_frame': 40000, 'header_fps': 25.,
                                'total_frames': 100, 'width': 640, 'height': 480}
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match='Truncated'):
        avi_header(path)


def test_projection_checks_denominator_and_does_not_assign_units():
    np.testing.assert_array_equal(project_points([[1, 2]], np.diag([2, 3, 1])), [[2, 6]])
    with pytest.raises(ValueError, match='undefined'):
        project_points([[1, 2]], np.diag([2, 3, 0]))


def test_source_forward_velocity_columns_are_never_loaded(tmp_path):
    path = tmp_path / 'obsmat.txt'
    path.write_text('10 1 2 0 3 9999 0 -9999\n20 1 4 0 5 nan nan nan\n')
    points, skipped = read_track(path)
    assert skipped == 0
    np.testing.assert_array_equal(points, [[10, 1, 2, 3], [20, 1, 4, 5]])


def test_row_identity_alignment_accounts_for_unmatched_source_rows():
    pixel = np.array([[10, 1, 2, 3], [20, 1, 4, 5], [30, 1, 6, 7]], float)
    aligned, report = align_source_rows(pixel, pixel[[1, 0]])
    np.testing.assert_array_equal(aligned, pixel[[1, 0]])
    assert report['pixel_only_frame_agent_keys'] == [[30., 1.]]
    with pytest.raises(ValueError, match='Duplicate'):
        align_source_rows(np.vstack([pixel, pixel[0]]), pixel)
    with pytest.raises(ValueError, match='lacks source'):
        align_source_rows(pixel[:2], pixel)


def test_header_fps_does_not_silently_override_annotation_clock():
    eth = clock_status(np.array([6]), np.array([100]), 25., .4)
    assert eth['documented_interval_header_step_conflict']
    assert eth['raw50_effective_seconds'] is None
    hotel = clock_status(np.array([10, 20]), np.array([100, 1]), 25., .4)
    assert not hotel['documented_interval_header_step_conflict']
    assert not hotel['effective_seconds_verified']
