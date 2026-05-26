from pathlib import Path

from src import stage42_calibration_evidence_refresh as s42ad


def test_stage42ad_parse_homography_like_matrix() -> None:
    text = """
    1.0 0.0 12.5
    0.0 1.0 -3.0
    0.0 0.0 1.0
    """
    matrices = s42ad.parse_homography_like_matrices(text)
    assert len(matrices) == 1
    assert matrices[0]["determinant"] == 1.0


def test_stage42ad_scan_file_finds_fps_stride_scale(tmp_path: Path) -> None:
    path = tmp_path / "calibration.txt"
    path.write_text(
        "Frame rate 10fps\nannotation stride 1\nmeters per pixel scale 0.05\n",
        encoding="utf-8",
    )
    scan = s42ad.scan_evidence_file(path)
    assert scan["fps_terms"]
    assert scan["stride_terms"]
    assert scan["scale_terms"]


def test_stage42ad_claim_status_blocks_pedestrian_metric() -> None:
    evidence = {
        "parseable_homography_matrix_count": 2,
        "fps_or_frame_rate_evidence_count": 1,
        "stride_or_dt_evidence_count": 1,
        "scale_or_meter_evidence_count": 0,
        "coordinate_unit_evidence": {},
        "metadata_metric_true_count": 0,
    }
    status = s42ad._claim_status("eth_ucy", evidence)
    assert status["metric_claim_status"] == "weak_metric_candidate_requires_manual_validation"
    assert status["seconds_claim_status"] == "effective_seconds_candidate_requires_manual_validation"
    assert status["metric_claim_allowed_for_pedestrian_world_model"] is False
    assert status["seconds_claim_allowed_for_official_pedestrian"] is False


def test_stage42ad_gate_requires_no_overclaim() -> None:
    datasets = []
    for dataset_id in ("sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"):
        datasets.append(
            {
                "dataset_id": dataset_id,
                "evidence_files_scanned": 1,
                "parseable_homography_matrix_count": 1 if dataset_id == "eth_ucy" else 0,
                "metric_claim_allowed_for_pedestrian_world_model": False,
                "seconds_claim_allowed_for_official_pedestrian": False,
                "metric_claim_status": "traffic_metric_diagnostic_only" if dataset_id == "tgsim" else "not_allowed",
            }
        )
    payload = {
        "datasets": datasets,
        "summary": {"global_metric_claim_allowed": False},
        "user_action_required": [{"dataset_id": "eth_ucy"}],
        "claim_boundary": {"stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42ad._gate(payload)
    assert gate["passed"] == gate["total"]
