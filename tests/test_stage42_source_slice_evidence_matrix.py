import numpy as np

from src import stage42_source_slice_evidence_matrix as jv


def test_source_slice_gate_passes_for_current_cache() -> None:
    payload = jv.run_stage42_source_slice_evidence_matrix(refresh_readmes=False)
    gate = payload["stage42_jv_gate"]
    assert gate["verdict"] == "stage42_jv_source_slice_evidence_matrix_pass"
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["domain_count"] >= 2
    assert payload["summary"]["all"]["positive_ade"] is True


def test_domain_horizon_metrics_include_t50_and_t100() -> None:
    payload = jv.run_stage42_source_slice_evidence_matrix(refresh_readmes=False)
    for domain, rows in payload["domain_horizon_metrics"].items():
        assert "50" in rows, domain
        assert "100" in rows, domain
        assert rows["50"]["rows"] > 0
        assert rows["100"]["rows"] > 0


def test_metric_detects_easy_degradation_when_selected_is_worse() -> None:
    cache = {
        "floor_ade": np.asarray([1.0, 1.0], dtype=np.float32),
        "selected_ade_seed_mean": np.asarray([1.2, 0.9], dtype=np.float32),
        "floor_fde": np.asarray([2.0, 2.0], dtype=np.float32),
        "selected_fde_seed_mean": np.asarray([2.0, 1.0], dtype=np.float32),
        "waypoint_valid": np.ones((2, 4), dtype=bool),
        "switch_any": np.asarray([True, False]),
    }
    row = jv._metric(cache, np.asarray([True, True]), "toy")
    assert row["easy_degradation"] > 0
    assert row["switch_rate"] == 0.5


def test_claim_boundary_blocks_metric_and_stage5c() -> None:
    payload = jv.run_stage42_source_slice_evidence_matrix(refresh_readmes=False)
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False
    assert payload["claim_boundary"]["true_3d"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
