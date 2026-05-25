from src.stage41_group_consistency_multiseed_repair import _candidate_eligible, _safety_buffer_score


def test_candidate_eligible_enforces_collision_ceiling() -> None:
    metrics = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "collision_delta_vs_floor_005": 0.006,
    }
    assert not _candidate_eligible(metrics, switch_rate=0.1, collision_ceiling=0.005)
    assert _candidate_eligible(metrics, switch_rate=0.1, collision_ceiling=0.01)


def test_safety_buffer_score_penalizes_switch_rate() -> None:
    base = {
        "metrics": {
            "all_improvement": 0.1,
            "t50_improvement": 0.1,
            "t100_improvement": 0.1,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.0,
            "collision_delta_vs_floor_005": 0.0,
        }
    }
    low_switch = {**base, "switch_rate": 0.1}
    high_switch = {**base, "switch_rate": 0.5}
    assert _safety_buffer_score(low_switch, 0.005) > _safety_buffer_score(high_switch, 0.005)
