from __future__ import annotations

from src.stage43_latent_state_dataset_contract import main


_PAYLOAD = None


def _payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = main()
    return _PAYLOAD


def test_stage43_latent_dataset_contract_passes_for_endpoint_training():
    payload = _payload()
    gate = payload["stage43_b_gate"]
    assert gate["verdict"] == "stage43_b_latent_state_dataset_contract_pass"
    assert gate["passed"] == gate["total"]
    assert gate["endpoint_latent_state_training_ready"] is True


def test_stage43_latent_dataset_rows_align_across_modal_artifacts():
    payload = _payload()
    for split, summary in payload["splits"].items():
        assert summary["row_alignment_pass"], split
        assert summary["rows"] > 0
        assert summary["history_k_available"]["8"] > 0
        assert summary["files"]["goal_prototypes"]["exists"]
        assert summary["files"]["baseline_family"]["exists"]


def test_stage43_latent_dataset_keeps_future_labels_out_of_inputs():
    payload = _payload()
    schema = payload["token_schema"]
    inputs = []
    for token, fields in schema.items():
        if token != "labels_only":
            inputs.extend(fields)
    assert "future_endpoint_x" not in inputs
    assert "future_endpoint_y" not in inputs
    assert "full_waypoint_xy_partial" not in inputs
    assert "waypoint_valid_partial" not in inputs
    assert payload["no_leakage"]["future_endpoint_input"] is False
    assert payload["no_leakage"]["future_waypoint_input"] is False
    assert payload["no_leakage"]["test_endpoint_goal_construction"] is False


def test_stage43_latent_dataset_records_full_waypoint_blocker():
    payload = _payload()
    full = payload["label_availability"]["full_waypoint"]
    assert full["status"] == "partial_eval_cache"
    assert full["train_val_supervised_full_waypoint_ready"] is False
    assert payload["stage43_b_gate"]["full_waypoint_supervised_training_ready"] is False
