import numpy as np

from src import stage42_sequence_full_waypoint as s42i


def test_stage42i_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42i.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42i_variant_masks_history_and_neighbor() -> None:
    split = {
        "agent_tokens": np.ones((2, 3, 4, 9), dtype=np.float32),
        "agent_mask": np.ones((2, 3), dtype=bool),
        "static": np.ones((2, 5), dtype=np.float32),
    }
    tokens, mask, static = s42i._variant_inputs(split, "sequence_waypoint_no_history")
    assert not np.any(tokens)
    assert not np.any(mask)
    assert np.all(static == 1.0)
    tokens, mask, _static = s42i._variant_inputs(split, "sequence_waypoint_no_neighbor")
    assert np.all(tokens[:, 0] == 1.0)
    assert not np.any(tokens[:, 1:])
    assert np.all(mask[:, 0])
    assert not np.any(mask[:, 1:])


def test_stage42i_gate_accepts_positive_sequence_full_waypoint() -> None:
    summary = {
        "sequence_waypoint_full": {
            "seeds": [53, 59, 61],
            "ade_all": {"mean": 0.1},
            "ade_t50": {"mean": 0.1},
            "ade_hard_failure": {"mean": 0.1},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "sequence_waypoint_no_history": {
            "seeds": [53, 59, 61],
            "ade_all": {"mean": 0.05},
            "ade_t50": {"mean": 0.02},
            "ade_hard_failure": {"mean": 0.04},
        },
        "sequence_waypoint_no_neighbor": {
            "seeds": [53, 59, 61],
            "ade_all": {"mean": 0.09},
            "ade_t50": {"mean": 0.08},
            "ade_hard_failure": {"mean": 0.07},
        },
    }
    result = {
        "dataset_rows": {"test": 10},
        "summary": summary,
        "contribution_vs_full": {
            "sequence_waypoint_no_history": {
                "ade_t50_delta_full_minus_ablation": 0.08,
                "ade_hard_delta_full_minus_ablation": 0.06,
            }
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42i._gate(result)
    assert gate["passed"] == gate["total"]
