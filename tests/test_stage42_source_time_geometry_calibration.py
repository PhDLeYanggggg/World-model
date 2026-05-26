from pathlib import Path

from src import stage42_source_time_geometry_calibration as bn


def test_parse_eth_homography_is_non_singular() -> None:
    row = bn.parse_homography_matrix(Path("external_data/OpenTraj/datasets/ETH/seq_eth/H.txt"))
    assert row["exists"] is True
    assert row["parseable"] is True
    assert abs(row["determinant"]) > 1e-12


def test_eth_source_has_local_meter_time_evidence() -> None:
    source = bn._audit_source(bn.SOURCE_SPECS[0])
    assert source["source_id"] == "ETH_seq_eth"
    assert source["coordinate"]["meter_coordinates_evidence"] is True
    assert source["timing"]["annotation_fps"] == 2.5
    assert source["timing"]["annotation_timestep_seconds"] == 0.4
    assert source["source_specific_metric_time_evidence"] is True
    assert source["global_metric_claim_allowed"] is False


def test_sdd_estimated_scale_is_not_metric_claim() -> None:
    row = bn._audit_sdd()
    assert row["scale_count"] > 0
    assert row["estimated_scale_warning_present"] is True
    assert row["metric_claim_allowed"] is False
    assert row["seconds_claim_allowed"] is False


def test_bn_gate_passes_but_blocks_global_claims() -> None:
    payload = bn.run_stage42_source_time_geometry_calibration()
    gate = payload["stage42_bn_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["source_specific_metric_time_sources"]
    assert payload["claim_boundary"]["global_metric_claim_allowed"] is False
    assert payload["claim_boundary"]["global_seconds_claim_allowed"] is False
    assert payload["claim_boundary"]["m3w_official_metric_seconds_claim_allowed"] is False
