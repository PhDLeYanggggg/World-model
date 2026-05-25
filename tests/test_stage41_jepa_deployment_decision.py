from src.stage41_jepa_deployment_decision import _stage18_evidence, _stage40_evidence


def test_stage18_evidence_negative_failure_lift() -> None:
    row = _stage18_evidence(
        {
            "non_collapse": True,
            "probe": {"no_jepa_failure_auc": 0.65, "jepa_frozen_failure_auc": 0.63},
        }
    )
    assert row["non_collapse"] is True
    assert row["deployable_lift"] is False
    assert row["downstream_lifts"]["failure_auroc_lift"] < 0


def test_stage40_evidence_requires_positive_and_safe_metrics() -> None:
    rows = _stage40_evidence(
        {
            "trials": {
                "jepa_aux_candidate_ranker": {
                    "test_metrics": {
                        "all_improvement": 0.1,
                        "t50_improvement": 0.1,
                        "hard_failure_improvement": 0.1,
                        "easy_degradation": 0.03,
                    }
                },
                "hybrid_moe_deeper_ranker": {
                    "test_metrics": {
                        "all_improvement": 0.1,
                        "t50_improvement": 0.1,
                        "hard_failure_improvement": 0.1,
                        "easy_degradation": 0.0,
                    }
                },
            }
        }
    )
    assert rows[0]["deployable_lift"] is False
    assert rows[1]["deployable_lift"] is True
