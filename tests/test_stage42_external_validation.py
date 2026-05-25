import numpy as np

from src import stage42_external_validation as s42


def test_stage42_external_validation_blocks_overclaim() -> None:
    text = "\n".join(s42.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_agent_key_parser_is_past_only_metadata() -> None:
    assert s42._agent_from_key("/tmp/source.txt|27|50") == "27"
    assert s42._agent_from_key("bad") == "unknown"


def test_metrics_easy_degradation_sign() -> None:
    floor = np.asarray([1.0, 1.0, 1.0, 1.0])
    selected = np.asarray([0.5, 1.5, 0.5, 1.5])
    labels = {
        "horizon": np.asarray([50, 50, 100, 100]),
        "hard": np.asarray([True, False, True, False]),
        "failure": np.asarray([False, False, False, False]),
        "easy": np.asarray([False, True, False, True]),
    }
    switch = np.asarray([True, True, True, True])
    metrics = s42._metrics(selected, floor, labels, switch)
    assert metrics["all_improvement"] == 0.0
    assert metrics["hard_failure_improvement"] > 0.0
    assert metrics["easy_degradation"] > 0.0


def test_stage42_gate_requires_ungated_safety_diagnosis() -> None:
    result = {
        "source_split": {
            "proposed_split_no_source_overlap": True,
            "frozen_model_eval_pool": {"source_fold_count": 3},
        },
        "comparisons": {
            "strongest_causal_baseline_or_stage37_floor": {},
            "teacher_repair_floor": {},
            "m3w_neural_v1_composite_tail_protected": {
                "all_improvement": 0.1,
                "t50_improvement": 0.1,
                "hard_failure_improvement": 0.1,
                "easy_degradation": 0.0,
                "by_domain": {"A": {}},
                "by_scene_top": [{}],
                "by_agent_top": [{}],
            },
            "ungated_neural_endpoint": {"easy_degradation": 0.5},
            "oracle_floor_vs_neural_diagnostic": {},
        },
        "cached_verified_comparisons": {"domain_local": {}},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42._gate(result)
    assert gate["passed"] == gate["total"]
