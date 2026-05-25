from src import stage41_neural_architecture_ablation_audit as audit


def test_deployable_requires_core_positive_and_two_domains():
    row = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.01,
        "by_domain": {
            "a": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0},
            "b": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0},
        },
    }
    assert audit._deployable(row)
    row["easy_degradation"] = 0.03
    assert not audit._deployable(row)


def test_safe_fallback_only_status():
    row = {
        "all_improvement": 0.0,
        "t50_improvement": 0.0,
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.0,
        "switch_rate": 0.0,
    }
    assert audit._safe_fallback_only(row)
    assert audit._status(row) == "safe_fallback_only_no_lift"


def test_best_candidate_prefers_deployable_over_negative():
    comparisons = {
        "bad": {"all_improvement": -1.0, "t50_improvement": -1.0, "hard_failure_improvement": -1.0, "easy_degradation": 0.0},
        "good": {
            "all_improvement": 0.1,
            "t50_improvement": 0.1,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.0,
            "by_domain": {
                "a": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0},
                "b": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0},
            },
        },
    }
    name, row = audit._best_candidate(comparisons, ["bad", "good"])
    assert name == "good"
    assert audit._deployable(row)
