from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from src import stage44_worldcore as wc


def test_token_feature_groups_keep_baseline_optional() -> None:
    names = [
        "current_x_over_scale",
        "history_speed_tail0",
        "prototype_distance_0",
        "scene_proxy::route_density",
        "graph_current_degree",
        "horizon_50",
        "baseline_endpoint_rel_0",
        "floor_endpoint_rel_x",
    ]
    groups = wc._token_feature_groups(names)
    assert groups["agent_state"] == [0]
    assert groups["agent_history"] == [1]
    assert groups["goal_prototype"] == [2]
    assert groups["walkable_obstacle"] == [3]
    assert groups["interaction_edge"] == [4]
    assert groups["time_source_domain_horizon"] == [5]
    assert groups["baseline_rollout"] == [6, 7]
    schema = wc._schema_from_groups(names, groups)
    assert schema["baseline_rollout"]["role"] == "optional_context"
    assert schema["latent_world_state"]["components"] == wc.LATENT_COMPONENTS


def test_worldcore_forward_shapes_and_no_baseline_flag() -> None:
    dims = {token: 3 for token in wc.TOKEN_TYPES}
    model = wc.WorldCoreModel(
        dims,
        hidden_dim=16,
        latent_dim=8,
        include_baseline=False,
        include_scene=True,
        include_interaction=True,
        use_jepa=True,
        use_transformer=True,
    )
    tokens = {token: torch.randn(4, 3) for token in wc.TOKEN_TYPES}
    target = torch.randn(4, 14)
    out = model(tokens, target)
    assert out["waypoint_delta"].shape == (4, 4, 2)
    assert out["z_next"].shape == (4, 8)
    assert out["future_world_latent"].shape == (4, 8)
    assert set(out["latent_state"].keys()) == set(wc.LATENT_COMPONENTS)
    assert out["goal_logits"].shape == (4, 8)
    assert out["validity_logits"].shape == (4, 2)
    assert model.include_baseline is False


def _metric(all_: float, t50: float, hard: float, easy: float) -> dict:
    return {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": all_,
        "endpoint_fde_improvement_vs_floor": all_,
        "t50_full_waypoint_ade_improvement_vs_floor": t50,
        "t50_endpoint_fde_improvement_vs_floor": t50,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": hard,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.1,
        "harm_over_floor_ade": -0.1,
        "mean_floor_ade": 1.0,
        "mean_selected_ade": 0.9,
    }


def _row(all_: float, t50: float, hard: float, easy: float = 0.0) -> dict:
    return {
        "config": {"use_jepa": True},
        "checkpoint_committed": False,
        "training_history": [{"epoch": 1}],
        "validation_policy": {"policy": {"gain_threshold": 0.5}},
        "test_eval": {
            "protected": _metric(all_, t50, hard, easy),
            "unprotected": _metric(all_, t50, hard, easy),
            "density_mse": 0.1,
            "interaction_auc": 0.7,
            "failure_auc": 0.7,
            "easy_auc": 0.7,
            "goal_direction_acc": 0.3,
            "physical_validity_proxy_acc": 0.9,
            "latent_variance": 0.2,
        },
    }


def test_gate_passes_when_worldcore_has_safe_lift() -> None:
    variants = {
        "no_baseline_latent": _row(-0.1, -0.1, -0.1),
        "baseline_aware_protected": _row(0.01, 0.0, 0.0),
        "hybrid_jepa_transformer": _row(0.03, 0.02, 0.04),
        "hybrid_no_scene": _row(0.01, 0.00, 0.02),
        "hybrid_no_interaction": _row(0.01, 0.00, 0.02),
        "hybrid_no_jepa": _row(0.01, 0.00, 0.02),
        "hybrid_no_transformer_ssm": _row(0.01, 0.00, 0.02),
    }
    payload = {
        "token_schema": {"tokens": {token: {} for token in wc.TOKEN_TYPES}, "latent_world_state": {"components": wc.LATENT_COMPONENTS}},
        "variants": variants,
        "best_variant": "hybrid_jepa_transformer",
        "ablation_table": {
            "hybrid_no_scene": {"all_delta_vs_hybrid": 0.02, "t50_delta_vs_hybrid": 0.02, "hard_delta_vs_hybrid": 0.02, "interaction_auc_delta_vs_hybrid": 0.0},
            "hybrid_no_interaction": {"all_delta_vs_hybrid": 0.02, "t50_delta_vs_hybrid": 0.02, "hard_delta_vs_hybrid": 0.02, "interaction_auc_delta_vs_hybrid": 0.1},
            "hybrid_no_jepa": {"all_delta_vs_hybrid": 0.02, "t50_delta_vs_hybrid": 0.02, "hard_delta_vs_hybrid": 0.02, "interaction_auc_delta_vs_hybrid": 0.0},
            "hybrid_no_transformer_ssm": {"all_delta_vs_hybrid": 0.02, "t50_delta_vs_hybrid": 0.02, "hard_delta_vs_hybrid": 0.02, "interaction_auc_delta_vs_hybrid": 0.0},
        },
        "failure_analysis": {"next_repairs": [], "repair_actions_executed": False, "jepa_downstream_lift": True, "interaction_lift": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    payload["lineage_preflight"] = {"training_allowed_with_unchanged_legacy_caches": True}
    payload["selection"] = {"protocol": "validation_only_before_test_prediction"}
    gate = wc._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["worldcore_lift_measured"] is True
    assert gate["verdict"] == "stage44_worldcore_latent_state_candidate_pass"
    assert gate["submission_ready"] is False
    del payload["lineage_preflight"]
    assert not wc._gate(payload)["gates"]["recording_teacher_boundary_audited"]


def test_gate_diagnostic_when_easy_is_not_safe() -> None:
    variants = {
        "no_baseline_latent": _row(-0.1, -0.1, -0.1),
        "baseline_aware_protected": _row(0.01, 0.0, 0.0, easy=0.03),
        "hybrid_jepa_transformer": _row(0.03, 0.02, 0.04, easy=0.03),
        "hybrid_no_scene": _row(0.01, 0.00, 0.02),
        "hybrid_no_interaction": _row(0.01, 0.00, 0.02),
        "hybrid_no_jepa": _row(0.01, 0.00, 0.02),
        "hybrid_no_transformer_ssm": _row(0.01, 0.00, 0.02),
    }
    payload = {
        "token_schema": {"tokens": {token: {} for token in wc.TOKEN_TYPES}, "latent_world_state": {"components": wc.LATENT_COMPONENTS}},
        "variants": variants,
        "best_variant": "hybrid_jepa_transformer",
        "ablation_table": {"hybrid_no_scene": {}, "hybrid_no_interaction": {}, "hybrid_no_jepa": {}, "hybrid_no_transformer_ssm": {}},
        "failure_analysis": {"next_repairs": ["tighten safety"], "repair_actions_executed": True, "jepa_downstream_lift": False, "interaction_lift": False},
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = wc._gate(payload)
    assert gate["gates"]["easy_preservation_safe"] is False
    assert gate["verdict"] == "stage44_worldcore_diagnostic_not_yet_independent_world_model"


def _validation_row(score: float, easy: float = 0.0) -> dict:
    row = _row(score, score, score, easy)
    row["validation_policy"] = {"policy": {}, "metrics": _metric(score, score, score, easy),
                                "objective": score}
    return row


def test_variant_selection_and_repair_ignore_test_metrics() -> None:
    rows = {"hybrid_jepa_transformer": _validation_row(0.1), "other": _validation_row(0.2)}
    before = wc._best_variant(rows)[0], wc._needs_repair(rows)
    changed = deepcopy(rows)
    changed["hybrid_jepa_transformer"]["test_eval"]["protected"] = _metric(1e6, 1e6, 1e6, 0)
    changed["other"]["test_eval"]["protected"] = _metric(-1e6, -1e6, -1e6, 100)
    assert (wc._best_variant(changed)[0], wc._needs_repair(changed)) == before
    assert before == ("other", False)


def test_selection_works_before_any_test_evaluation() -> None:
    rows = {"hybrid_jepa_transformer": _validation_row(0.1)}
    del rows["hybrid_jepa_transformer"]["test_eval"]
    assert wc._best_variant(rows)[0] == "hybrid_jepa_transformer"
    assert not wc._needs_repair(rows)


def test_missing_validation_never_falls_back_to_test() -> None:
    with pytest.raises(ValueError, match="validation"):
        wc._best_variant({"test_only": _row(1, 1, 1)})


def test_unsafe_validation_or_nonfinite_objective_is_not_selected() -> None:
    rows = {"safe": _validation_row(0.01), "unsafe": _validation_row(5.0, 0.03),
            "nan": _validation_row(float("nan"))}
    assert wc._best_variant(rows)[0] == "safe"


def test_no_safe_predicted_gain_keeps_validation_floor(monkeypatch) -> None:
    from types import SimpleNamespace
    base = SimpleNamespace(x=np.zeros((2, 1)), floor_ade=np.ones(2), floor_fde=np.ones(2))
    val = SimpleNamespace(base=base)
    monkeypatch.setattr(wc.m, "_select_with_policy", lambda *_: (np.ones(2) * 2, np.ones(2) * 2, np.ones(2, bool)))
    def metrics(_base, ade, _fde, switched):
        return {**_metric(1 - float(ade.mean()), -1 if switched.any() else 0,
                          -1 if switched.any() else 0, 0), "switch_rate": float(switched.mean())}
    monkeypatch.setattr(wc.m, "_metrics", metrics)
    result = wc._search_policy(val, {})
    assert result["objective"] == 0.0
    assert result["metrics"]["switch_rate"] == 0


def test_lineage_preflight_blocks_before_training_or_output(monkeypatch) -> None:
    from types import SimpleNamespace
    def blocked():
        raise RuntimeError("lineage preflight failed")
    def forbidden(*_args, **_kwargs):
        pytest.fail("A blocked run must not train or touch old artifacts")
    monkeypatch.setattr(wc, "assert_legacy_cache_safe", blocked)
    monkeypatch.setattr(wc, "ensure_dir", forbidden)
    monkeypatch.setattr(wc, "_train_variant", forbidden)
    with pytest.raises(RuntimeError, match="lineage"):
        wc._run(SimpleNamespace())


def test_training_and_selection_finish_before_test_is_loaded(monkeypatch):
    from types import SimpleNamespace
    events = []
    monkeypatch.setattr(wc, "assert_legacy_cache_safe", lambda: {"training_allowed_with_unchanged_legacy_caches": True})
    monkeypatch.setattr(wc, "ensure_dir", lambda *_: None)
    monkeypatch.setattr(wc.m, "_configure_runtime", lambda *_: None)
    monkeypatch.setattr(wc, "_build_worldcore_development_splits",
                        lambda *_: (object(), object(), {"token_schema": {}}, (np.zeros(1), np.ones(1))))
    def train(cfg, *_args, **_kwargs):
        assert "test_loaded" not in events
        events.append("trained_" + cfg.name)
        row = _validation_row(0.1)
        del row["test_eval"]
        row["checkpoint_sha256"] = "synthetic"
        return row
    monkeypatch.setattr(wc, "_train_variant", train)
    def write(path, payload):
        if path.name == "validation_selection_lock.json":
            assert payload["protocol"] == "validation_only_before_test_prediction"
            events.append("locked")
    monkeypatch.setattr(wc, "write_json", write)
    def test_split(*_):
        assert events[-1] == "locked"
        assert sum(e.startswith("trained_") for e in events) == 7
        events.append("test_loaded")
        return object(), {}
    monkeypatch.setattr(wc, "_build_worldcore_test_split", test_split)
    monkeypatch.setattr(wc, "_evaluate_frozen_variants", lambda *_: events.append("test_evaluated"))
    monkeypatch.setattr(wc, "_ablation_table", lambda *_: {})
    monkeypatch.setattr(wc, "_failure_analysis", lambda *_: {})
    monkeypatch.setattr(wc, "_gate", lambda *_: {})
    monkeypatch.setattr(wc, "_write_outputs", lambda *_: None)
    monkeypatch.setattr(wc, "_combined_hash", lambda *_: "synthetic")
    result = wc._run(SimpleNamespace(quick=True, medium=False, seed=1, batch_size=2))
    assert events[-2:] == ["test_loaded", "test_evaluated"]
    assert result["selection"]["confirmatory_evidence"] is False
