from src import stage41_t50_rescue as rescue


def test_stage41_t50_rescue_removes_t50_slices() -> None:
    policy = {"slices": {"ETH_UCY|10": {"a": 1}, "ETH_UCY|50": {"b": 2}, "UCY|100": {"c": 3}}}
    out = rescue._policy_without_t50(policy)
    assert "ETH_UCY|50" not in out["slices"]
    assert "ETH_UCY|10" in out["slices"]
    assert "UCY|100" in out["slices"]


def test_stage41_t50_rescue_score_prioritizes_t50() -> None:
    good = {"t50_improvement": 0.1, "all_improvement": 0.02, "hard_failure_improvement": 0.02, "t100_improvement": 0.0, "easy_degradation": 0.0}
    bad = dict(good)
    bad["t50_improvement"] = -0.1
    assert rescue._score_for_rescue(good) > rescue._score_for_rescue(bad)
