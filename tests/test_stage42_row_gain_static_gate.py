import numpy as np

from src import stage42_row_gain_static_gate as s42n


def test_stage42n_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42n.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42n_alpha_teacher_prefers_static_only_when_row_gain_positive() -> None:
    floor = np.asarray([5.0, 5.0, 5.0], dtype=np.float32)
    # rows: alpha .25 helps, no-static best, easy row static would help but is blocked
    alpha_ade = np.asarray(
        [
            [4.0, 3.0, 4.0],
            [3.5, 3.2, 3.5],
            [3.8, 3.5, 3.6],
            [4.1, 3.7, 3.8],
            [4.2, 4.0, 4.2],
        ],
        dtype=np.float32,
    )
    teacher, static_gain, floor_gain, harm = s42n._alpha_teacher_from_ade(
        floor,
        alpha_ade,
        np.asarray([False, False, True]),
    )
    assert teacher.tolist() == [0.25, 0.0, 0.0]
    assert static_gain[0] > 0
    assert floor_gain[0] > 0
    assert float(harm[0]) == 0.0


def test_stage42n_gate_requires_train_val_teacher_and_t50_gain() -> None:
    base = {
        "teacher_diagnostics": {"train": {"static_positive_rate": 0.2}},
        "rows": [{"seed": 109}, {"seed": 113}, {"seed": 127}],
        "summary": {
            "seeds": [109, 113, 127],
            "ade_all": {"mean": 0.03},
            "ade_t50": {"mean": 0.01},
            "ade_hard_failure": {"mean": 0.03},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "comparison": {
            "stage42_m_policy_distilled": {"ade_t50": {"mean": -0.001}},
            "stage42_l_horizon_static_gate": {"ade_all": {"mean": 0.02}, "ade_t50": {"mean": 0.002}},
        },
        "source_labels": {"row_teacher_test": "not_built"},
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
    assert s42n._gate(base)["passed"] == s42n._gate(base)["total"]
    base["source_labels"]["row_teacher_test"] = "fresh_run"
    assert s42n._gate(base)["gates"]["row_teacher_train_val_only"] is False
