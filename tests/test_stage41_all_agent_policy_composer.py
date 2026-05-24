import numpy as np

from src import stage41_all_agent_policy_composer as composer


def test_stage41_policy_composer_prefers_t50_when_configured() -> None:
    floor = np.array([10.0, 10.0, 10.0])
    horizon = np.array([10, 50, 50])
    risk_selected = np.array([9.0, 8.0, 8.0])
    risk_switch = np.array([True, True, True])
    t50_selected = np.array([10.0, 5.0, 6.0])
    t50_switch = np.array([False, True, True])
    selected, switch = composer._compose_arrays(
        "risk_non_t50_plus_t50",
        floor,
        horizon,
        risk_selected,
        risk_switch,
        t50_selected,
        t50_switch,
    )
    assert selected.tolist() == [9.0, 5.0, 6.0]
    assert switch.tolist() == [True, True, True]


def test_stage41_policy_composer_score_penalizes_easy_harm() -> None:
    safe = {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "t100_improvement": 0.1, "easy_degradation": 0.0}
    unsafe = dict(safe)
    unsafe["easy_degradation"] = 0.2
    assert composer._score(safe) > composer._score(unsafe)
