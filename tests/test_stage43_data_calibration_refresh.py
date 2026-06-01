from __future__ import annotations

from src import stage43_data_calibration_refresh as stage43_as


def test_stage43_specs_include_aerialmpt_local_zip() -> None:
    specs = {row["id"]: row for row in stage43_as._fresh_specs()}
    assert "aerialmpt" in specs
    assert "data/aerialmpt/DLR_AerialMPT_Dataset.zip" in specs["aerialmpt"]["raw_candidates"]


def test_source_specific_calibration_blocks_global_claim() -> None:
    payload = {
        "summary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "m3w_official_metric_seconds_claim_allowed": False,
        },
        "source_records": [
            {"source_id": "ETH_seq_eth", "domain": "ETH_UCY", "source_specific_metric_time_evidence": True},
            {"source_id": "UCY_zara01", "domain": "UCY", "source_specific_metric_time_evidence": True},
            {"source_id": "TrajNet", "domain": "TrajNet", "source_specific_metric_time_evidence": False},
        ],
    }
    result = stage43_as._source_specific_calibration(payload)
    assert result["supported_source_count"] == 2
    assert result["supported_by_domain"] == {"ETH_UCY": 1, "UCY": 1}
    assert result["global_metric_claim_allowed"] is False
    assert result["global_seconds_claim_allowed"] is False


def _payload(*, global_metric: bool = False, calibration_count: int = 6) -> dict:
    datasets = {
        dataset_id: {
            "dataset_id": dataset_id,
            "source": "fresh_run",
            "data_role": "diagnostic_only" if dataset_id == "tgsim" else "external_eval",
            "metric_claim_allowed": dataset_id == "tgsim",
            "seconds_claim_allowed": False,
        }
        for dataset_id in ["sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"]
    }
    return {
        "source": stage43_as.SOURCE,
        "datasets": list(datasets.values()),
        "summary": {
            "external_domains_ready_from_existing_state": ["opentraj", "eth_ucy", "trajnet", "ucy"],
            "global_metric_claim_allowed": global_metric,
            "global_seconds_claim_allowed": False,
        },
        "source_specific_calibration": {
            "supported_source_count": calibration_count,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
        },
        "sdd_status": {
            "coordinate_unit": "pixel",
            "metric_claim_allowed": False,
            "seconds_claim_allowed": False,
        },
        "training_run": False,
        "auto_download_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def test_gate_passes_for_raw_frame_guarded_calibration_refresh() -> None:
    gate = stage43_as._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["data_calibration_ready"] is True


def test_gate_fails_for_global_metric_overclaim() -> None:
    gate = stage43_as._gate(_payload(global_metric=True))
    assert gate["gates"]["global_metric_seconds_blocked"] is False
    assert gate["data_calibration_ready"] is False


def test_gate_fails_without_source_specific_calibration_records() -> None:
    gate = stage43_as._gate(_payload(calibration_count=0))
    assert gate["gates"]["source_specific_calibration_recorded"] is False
    assert gate["data_calibration_ready"] is False
