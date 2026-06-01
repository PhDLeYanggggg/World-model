from __future__ import annotations

from src import stage43_feature_family_multiseed_confirmation as ai


def _summary(name: str, *, t50_delta: float = 0.0, all_delta: float = 0.0, hard_delta: float = 0.0, pos: int = 3) -> dict:
    row = {
        "variant": name,
        "seeds": [431, 443, 457],
        "metrics": {
            "full_waypoint_ade_improvement_vs_floor": {"mean": 0.2, "std": 0.01, "min": 0.18, "max": 0.22, "positive_seed_count": 3},
            "t50_full_waypoint_ade_improvement_vs_floor": {"mean": 0.2, "std": 0.01, "min": 0.18, "max": 0.22, "positive_seed_count": 3},
            "hard_failure_full_waypoint_ade_improvement_vs_floor": {"mean": 0.2, "std": 0.01, "min": 0.18, "max": 0.22, "positive_seed_count": 3},
            "easy_degradation_vs_floor": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "positive_seed_count": 0},
        },
    }
    if name != "full_features":
        row["delta_full_minus_variant"] = {
            "full_waypoint_ade_improvement_vs_floor": {"mean": all_delta, "std": 0.01, "min": all_delta, "max": all_delta, "positive_seed_count": pos if all_delta > 0 else 0},
            "t50_full_waypoint_ade_improvement_vs_floor": {"mean": t50_delta, "std": 0.01, "min": t50_delta, "max": t50_delta, "positive_seed_count": pos if t50_delta > 0 else 0},
            "hard_failure_full_waypoint_ade_improvement_vs_floor": {"mean": hard_delta, "std": 0.01, "min": hard_delta, "max": hard_delta, "positive_seed_count": pos if hard_delta > 0 else 0},
        }
    return row


def _payload(*, stable: bool) -> dict:
    return {
        "source": ai.SOURCE,
        "result_source": "fresh_multiseed_retrained_feature_family_confirmation",
        "seeds": [431, 443, 457],
        "variants": ["full_features", "no_history", "no_goal", "no_baseline_floor", "no_domain"],
        "variant_summaries": [
            _summary("full_features"),
            _summary("no_history", t50_delta=0.02 if stable else 0.0),
            _summary("no_goal", hard_delta=0.03 if stable else 0.0),
            _summary("no_baseline_floor", t50_delta=0.12 if stable else 0.0),
            _summary("no_domain"),
        ],
        "stable_positive_t50_contribution_variants": ["no_history", "no_baseline_floor"] if stable else [],
        "stable_positive_hard_or_all_contribution_variants": ["no_goal"] if stable else [],
        "seed_results": [
            {
                "seed": 431,
                "variants": [
                    {"variant": "full_features", "checkpoint_committed": False},
                    {"variant": "no_history", "checkpoint_committed": False},
                    {"variant": "no_goal", "checkpoint_committed": False},
                    {"variant": "no_baseline_floor", "checkpoint_committed": False},
                    {"variant": "no_domain", "checkpoint_committed": False},
                ],
            }
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_requires_stable_multiseed_contributions() -> None:
    passing = ai._gate(_payload(stable=True))
    assert passing["passed"] == passing["total"]
    assert passing["multiseed_feature_family_contribution_supported"] is True

    failing = ai._gate(_payload(stable=False))
    assert failing["gates"]["baseline_floor_t50_contribution_stable"] is False
    assert failing["gates"]["at_least_two_stable_module_contributions"] is False
    assert failing["multiseed_feature_family_contribution_supported"] is False
