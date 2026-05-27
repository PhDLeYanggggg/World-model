from __future__ import annotations

from pathlib import Path

from src.stage42_raw_source_time_geometry_hint_audit import (
    _gate,
    _parse_frame_stride,
    _parse_h_matrix,
    _parse_time_metadata,
)


def test_parse_h_matrix_detects_non_singular_matrix() -> None:
    parsed = _parse_h_matrix("1 0 0\n0 2 0\n0 0 3\n")

    assert parsed is not None
    assert parsed["matrix"] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    assert parsed["determinant"] == 6.0
    assert parsed["non_singular_hint"] is True


def test_parse_time_metadata_extracts_annotation_fps_and_timestep() -> None:
    parsed = _parse_time_metadata(
        "The video was shot at 25 fps. Annotation was done at 2.5 fps "
        "with a timestep of 0.4 seconds."
    )

    assert parsed["fps_values"] == [25.0, 2.5]
    assert parsed["timestep_seconds_values"] == [0.4]
    assert parsed["annotation_fps_hint"] == 2.5


def test_parse_frame_stride_uses_first_numeric_column(tmp_path: Path) -> None:
    p = tmp_path / "tracks.txt"
    p.write_text(
        "\n".join(
            [
                "0 1 0.0 0.0",
                "0 2 1.0 1.0",
                "10 1 0.5 0.0",
                "20 1 1.0 0.0",
                "30 1 1.5 0.0",
            ]
        )
    )

    parsed = _parse_frame_stride(p)

    assert parsed is not None
    assert parsed["mode_frame_stride"] == 10.0
    assert parsed["unique_frames_sampled"] == 4


def test_gate_requires_no_metric_or_seconds_overclaim() -> None:
    payload = {
        "stage42_ds_verdict": "pass",
        "stage42_dt_verdict": "pass",
        "data_calibration_addendum_written": True,
        "summary": {
            "targets_checked": 7,
            "targets_with_h_matrix_hints": 2,
            "targets_with_time_hints": 1,
            "targets_with_frame_stride_hints": 3,
            "metric_time_subset_hint_targets": 1,
            "legal_conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "world_state_rows_generated": 0,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": True,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["gates"]["no_metric_seconds_overclaim"] is False
    assert gate["verdict"] == "stage42_du_raw_source_time_geometry_hint_audit_partial"
