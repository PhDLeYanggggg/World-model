from __future__ import annotations

from src.stage43_blocked_source_terms_identity_packet import (
    _biwi_packet,
    _gate,
    _links_from_text,
    _terms_packet_row,
)


def test_links_from_text_dedupes_and_strips_trailing_punctuation() -> None:
    text = "See https://example.org/data, and https://example.org/data. Also https://x.y/z)"
    assert _links_from_text(text) == ["https://example.org/data", "https://x.y/z"]


def test_terms_packet_row_keeps_conversion_and_training_blocked() -> None:
    row = _terms_packet_row(
        {
            "dataset_name": "Wild-Track",
            "local_path": "/tmp/wildtrack",
            "technical_support_candidate": True,
            "support_family": "TrajNet_mot",
            "point_rows": 100,
            "agent_tracks": 10,
            "t50_candidate_rows": 20,
            "t100_candidate_rows": 5,
            "calibration_file_count": 1,
            "coordinate_unit": "dataset_local",
            "metric_status": "unverified_weak_metric",
        }
    )
    assert row["technical_support_candidate"] is True
    assert row["conversion_ready_now"] is False
    assert row["guarded_conversion_allowed_now"] is False
    assert row["training_allowed_now"] is False
    assert "terms_not_confirmed_by_user" in row["blockers"]
    assert "source_identity_not_confirmed_by_user" in row["blockers"]
    assert "not_converted_into_stage43_feature_store" in row["blockers"]


def test_biwi_packet_requires_independent_source() -> None:
    packet = _biwi_packet(
        {
            "family_readiness": {
                "TrajNet_biwi": {
                    "status": "blocked_by_no_independent_source",
                    "technical_candidate_count": 1,
                    "conversion_ready_count": 0,
                }
            }
        }
    )
    assert packet["repair_training_allowed_now"] is False
    assert "independent_biwi_like_source_missing" in packet["blockers"]
    assert packet["manual_fields_required"]["heldout_source_disjoint_from_train_val"] is False


def test_gate_passes_for_terms_identity_packet_payload() -> None:
    packet = _terms_packet_row(
        {
            "dataset_name": "Wild-Track",
            "technical_support_candidate": True,
            "support_family": "TrajNet_mot",
            "t50_candidate_rows": 20,
            "t100_candidate_rows": 5,
            "calibration_file_count": 1,
        }
    )
    packet_2 = dict(packet, dataset_name="PETS-2009-S2L1")
    packet_3 = dict(packet, dataset_name="Town-Center")
    payload = {
        "input_verdicts": {
            "stage43_be": "stage43_be_blocked_source_support_acquisition_preflight_pass"
        },
        "summary": {
            "dataset_terms_packets": 3,
            "technical_candidates": 3,
            "official_hint_rows": 3,
            "manual_terms_required_rows": 3,
            "conversion_ready_now": 0,
            "guarded_conversion_allowed_now": 0,
            "training_allowed_now": 0,
            "biwi_independent_source_ready": False,
        },
        "dataset_packets": [packet, packet_2, packet_3],
        "biwi_independent_source_packet": {
            "repair_training_allowed_now": False,
        },
        "next_required_actions": ["a", "b", "c", "d"],
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "template_is_permission": False,
            "converted_external_support_source": False,
            "blocked_source_repair_success_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage43_bf_blocked_source_terms_identity_packet_pass"
    assert gate["passed"] == gate["total"]
