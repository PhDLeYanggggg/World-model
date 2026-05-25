import numpy as np

from src import stage41_bounded_neural_blend_dynamics as blend


def test_alpha_vector_supports_global_policy():
    data = {
        "labels": {"horizon": np.asarray([10, 50]), "domain": np.asarray(["A", "B"])},
        "teacher_raw_switch": np.asarray([True, False]),
        "teacher_repaired_switch": np.asarray([False, True]),
        "teacher_prob": np.asarray([0.8, 0.2]),
    }
    alpha = blend._alpha_vector(data, {"type": "global", "alpha": 0.1})
    assert np.allclose(alpha, [0.1, 0.1])
    gated = blend._alpha_vector(data, {"type": "global", "alpha": 0.1, "gate": "teacher_raw_switch"})
    assert np.allclose(gated, [0.1, 0.0])
    repaired = blend._alpha_vector(data, {"type": "global", "alpha": 0.1, "gate": "teacher_repaired_switch"})
    assert np.allclose(repaired, [0.0, 0.1])


def test_alpha_vector_supports_horizon_policy():
    data = {
        "labels": {"horizon": np.asarray([10, 25, 50, 100]), "domain": np.asarray(["A", "A", "A", "A"])},
        "teacher_raw_switch": np.asarray([True, True, True, True]),
        "teacher_repaired_switch": np.asarray([True, True, True, True]),
        "teacher_prob": np.asarray([0.1, 0.2, 0.3, 0.4]),
    }
    alpha = blend._alpha_vector(data, {"type": "horizon", "alpha_by_horizon": {10: 0.01, 25: 0.02, 50: 0.03, 100: 0.04}})
    assert np.allclose(alpha, [0.01, 0.02, 0.03, 0.04])


def test_eligible_requires_easy_and_collision_safety():
    metrics = {
        "all_improvement": 0.01,
        "t50_improvement": 0.01,
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.0,
        "collision_delta_vs_floor_005": 0.0,
        "alpha_mean": 0.1,
    }
    assert blend._eligible(metrics, 0.01)
    bad_easy = dict(metrics, easy_degradation=0.03)
    assert not blend._eligible(bad_easy, 0.01)
    bad_collision = dict(metrics, collision_delta_vs_floor_005=0.02)
    assert not blend._eligible(bad_collision, 0.01)


def test_safe_switch_policy_candidates_are_constrained_to_repaired_gate():
    policies = blend._safe_switch_candidate_policies()
    assert policies
    assert all(p.get("gate") == "teacher_repaired_switch" or p["type"] == "composite_tail" for p in policies)
    assert any(p.get("alpha") == 1.0 for p in policies if p["type"] == "global")
    assert any(p.get("alpha") == 1.4 for p in policies if p["type"] == "global")
    assert any(p["type"] == "composite_tail" for p in policies)


def test_composite_tail_adds_low_alpha_only_to_safe_tail_rows():
    data = {
        "labels": {"horizon": np.asarray([50, 50, 50]), "domain": np.asarray(["A", "A", "A"])},
        "teacher_repaired_switch": np.asarray([True, False, False]),
        "proposal_gain": np.asarray([0.0, 0.2, 0.2]),
        "proposal_harm": np.asarray([0.0, 0.1, 0.9]),
        "proposal_uncertainty": np.asarray([0.0, 0.1, 0.1]),
        "proposal_teacher_prob": np.asarray([1.0, 0.9, 0.9]),
    }
    alpha = blend._alpha_vector(
        data,
        {
            "type": "composite_tail",
            "switch_alpha": 1.0,
            "tail_alpha": 0.1,
            "tail_gain_min": 0.05,
            "tail_harm_max": 0.2,
            "tail_uncertainty_max": 0.2,
            "tail_teacher_min": 0.45,
        },
    )
    assert np.allclose(alpha, [1.0, 0.1, 0.0])
