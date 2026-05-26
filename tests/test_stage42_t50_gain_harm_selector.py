import numpy as np

from src import stage42_t50_gain_harm_selector as s42p


def test_stage42p_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42p.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42p_target_weights_t50_rows() -> None:
    teacher = {
        "floor_gain": np.asarray([0.1, 0.1, 0.0], dtype=np.float32),
        "harm": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "switchable": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "weight": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    }
    split = {"horizon": np.asarray([50, 10, 50], dtype=np.int64)}
    target = s42p._target_t50_weighted(teacher, split)
    assert target["weight"][0] > target["weight"][1]
    assert target["weight"][0] > target["weight"][2]


def test_stage42p_gate_requires_t50_specific_and_no_test_stats() -> None:
    base = {
        "rows": [{"seed": 149}, {"seed": 151}, {"seed": 157}],
        "summary": {
            "ade_all": {"mean": 0.05},
            "ade_t50": {"mean": 0.01},
            "ade_hard_failure": {"mean": 0.05},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "comparison": {"stage42_o_explicit_gain_harm_selector": {"ade_t50": {"mean": -0.001}}},
        "source_labels": {
            "t50_weighted_teacher": True,
            "row_teacher_test": "not_built",
            "policy_uses_easy_label": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42p._gate(base)
    assert gate["passed"] == gate["total"]
    base["no_leakage"]["test_statistics_normalization"] = True
    assert s42p._gate(base)["gates"]["no_test_statistics_normalization"] is False
