"""Source-time boundary regressions; no actual DroneCrowd labels used."""
import pytest

from scripts.audit_m3w_annotation_export_provenance import (
    interpolation_witness, source_bytes,
)
from src.evaluation.m3w_dronecrowd_intake import require_forecast_admission


def test_nominal_past_and_backward_velocity_can_still_depend_on_future_control():
    result = interpolation_witness()
    assert result['observed_frames'] == list(range(1, 9))
    assert result['all_nominal_input_frames_not_after_query'] is True
    assert result['source_control_after_query'] == 10 > result['query_frame']
    assert result['history_rows_changed_by_altering_future_control'] == 8
    assert result['last_causal_fd_velocity_before'] == [1., 0.]
    assert result['last_causal_fd_velocity_after'] == [2., 0.]


def test_unflagged_xml_cannot_identify_direct_versus_interpolated_source():
    result = interpolation_witness()
    assert result['same_unflagged_xml_from_direct_or_interpolated_rows'] is True
    assert result['structural_screen_status'] == 'structure_only_not_forecast_admission'
    assert result['structural_screen_interpolation_provenance'].startswith('unresolved')
    assert result['causal_rows_exported_by_screen'] == 0
    with pytest.raises(ValueError, match='cannot approve'):
        require_forecast_admission(result)


def test_source_code_is_not_silently_acquired_or_executed(tmp_path):
    def forbidden_fetch(*args):
        raise AssertionError('Network was not authorized')
    with pytest.raises(ValueError, match='explicit --download-source'):
        source_bytes(tmp_path, fetcher=forbidden_fetch)


def test_wrong_source_bytes_never_saved(tmp_path):
    with pytest.raises(ValueError, match='pinned Git blob'):
        source_bytes(tmp_path, True, fetcher=lambda *args: b'print("do not execute")')
    assert list(tmp_path.iterdir()) == []


def test_existing_wrong_size_source_refused(tmp_path):
    (tmp_path / 'README.md').write_text('not author source')
    with pytest.raises(ValueError, match='size changed'):
        source_bytes(tmp_path)
