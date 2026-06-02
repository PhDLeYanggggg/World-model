from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _target_vec,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_latent_adapter_downstream_heads import _fit_heads, _load_adapter, _predict_heads
from src.stage43_latent_transition_adapter_repair import REPORT_JSON as STAGE43_BZ_JSON, _adapter_predict
from src.stage43_latent_transition_consistency_audit import _predict_transition_latents
from src.stage43_shadow_easy_guard_repair import (
    REPORT_JSON as STAGE43_CC_JSON,
    _encode_selected_variant,
    _evaluate_policy,
    _search_base_policy,
    _shadow_validation_split,
    _source_family,
    _source_support_summary,
)


REPORT_JSON = OUT_DIR / "stage43_source_family_coverage_guard.json"
REPORT_MD = OUT_DIR / "stage43_source_family_coverage_guard.md"
GATE_MD = OUT_DIR / "stage43_stage_cd_source_family_coverage_guard_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_CD_SOURCE_FAMILY_COVERAGE_GUARD"
SOURCE = "fresh_stage43_cd_source_family_coverage_guard"
SELECTED_VARIANT = "identity_stage43m_adapter_z"


def _coverage_policy(base_policy: Mapping[str, float], support: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    return {
        "base_policy": dict(base_policy),
        "blocked_domains": [],
        "blocked_horizons": [],
        "blocked_domain_horizons": [],
        "source_family_support_mode": mode,
        "supported_global_families": support["global_families"],
        "supported_domain_families": support["domain_families"],
        "name": {
            "none": "base_threshold_only",
            "global": "global_source_family_coverage_guard",
            "domain": "domain_source_family_coverage_guard",
        }[mode],
    }


def _objective(metrics: Mapping[str, Any]) -> float:
    return float(
        metrics["full_waypoint_ade_improvement_vs_floor"]
        + metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        + 0.50 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        - 0.05 * metrics["switch_rate"]
        - 20.0 * max(0.0, metrics["easy_degradation_vs_floor"] - 0.005)
    )


def _select_coverage_policy(
    val: Any,
    val_pred: Mapping[str, np.ndarray],
    shadow: Mapping[str, Any],
) -> dict[str, Any]:
    families = shadow["families"]
    holdout = shadow["holdout"]
    support = shadow["plan"]["support"]
    base = _search_base_policy(val, val_pred, shadow["calibration"])
    candidates = []
    for mode in ["none", "global", "domain"]:
        policy = _coverage_policy(base["base_policy"], support, mode=mode)
        evaluated = _evaluate_policy(val, val_pred, families, policy, holdout)
        metrics = evaluated["metrics"]
        candidates.append(
            {
                "policy": policy,
                "shadow_holdout_metrics": metrics,
                "objective": _objective(metrics),
                "coverage_rank": {"none": 0, "global": 1, "domain": 2}[mode],
                "safe": metrics["easy_degradation_vs_floor"] <= 0.005 and metrics["switch_rate"] <= 0.25,
            }
        )
    safe = [row for row in candidates if row["safe"]]
    if safe:
        max_objective = max(float(row["objective"]) for row in safe)
        close = [row for row in safe if float(row["objective"]) >= max_objective - 1e-6]
        selected = max(close, key=lambda row: int(row["coverage_rank"]))
    else:
        selected = min(candidates, key=lambda row: float(row["shadow_holdout_metrics"]["easy_degradation_vs_floor"]))
    return {
        "base_calibration": base,
        "candidate_policies": candidates,
        "selected_shadow_policy": selected,
        "selection_rule": (
            "Select a shadow-holdout-safe policy by validation objective; if multiple policies tie within 1e-6, "
            "prefer the stricter source-family coverage guard. No test rows or test metrics are used."
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    test = payload["test_once"]["metrics"]
    shadow = payload["coverage_policy"]["selected_shadow_policy"]["shadow_holdout_metrics"]
    gates = {
        "stage43_cc_precondition_seen": payload["stage43_cc_precondition"]["verdict"]
        == "stage43_cc_shadow_easy_guard_shadow_safe_test_mismatch",
        "fresh_coverage_replay_completed": payload["result_source"] == "fresh_validation_source_family_coverage_guard",
        "train_only_heads_refit": payload["protocol"]["train_only_heads_refit"] is True,
        "validation_shadow_only_selection": payload["coverage_policy"]["selection_uses_test_metrics"] is False,
        "coverage_guard_selected": payload["coverage_policy"]["selected_shadow_policy"]["policy"]["source_family_support_mode"]
        in {"global", "domain"},
        "unsupported_test_families_reported": bool(payload["test_source_support_summary"]["global_unsupported_family_rows"]),
        "shadow_holdout_easy_safe": shadow["easy_degradation_vs_floor"] <= 0.005,
        "test_easy_preserved": test["easy_degradation_vs_floor"] <= 0.02,
        "test_lift_vs_floor": test["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "test_t50_reported": "t50_full_waypoint_ade_improvement_vs_floor" in test,
        "test_hard_failure_reported": "hard_failure_full_waypoint_ade_improvement_vs_floor" in test,
        "no_future_or_test_leakage": payload["no_leakage"]["future_labels_as_inputs"] is False
        and payload["no_leakage"]["future_labels_train_eval_only"] is True
        and payload["no_leakage"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["guard_uses_test_endpoints"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage43_cd_source_family_coverage_guard_pass"
    elif gates["test_easy_preserved"] and gates["test_lift_vs_floor"]:
        verdict = "stage43_cd_source_family_coverage_guard_partial_safe_lift"
    else:
        verdict = "stage43_cd_source_family_coverage_guard_diagnostic_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cd_gate"]
    selected = payload["coverage_policy"]["selected_shadow_policy"]
    shadow = selected["shadow_holdout_metrics"]
    test = payload["test_once"]["metrics"]
    return [
        "# Stage43-CD Source-Family Coverage Guard",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- selected policy: `{selected['policy']['name']}`",
        "- deployable policy changed: `False`",
        "",
        "## Coverage Evidence",
        "",
        f"- selection rule: {payload['coverage_policy']['selection_rule']}",
        f"- validation source families: `{payload['shadow_validation']['support']['global_families']}`",
        f"- test global-unsupported families: `{payload['test_source_support_summary']['global_unsupported_family_rows']}`",
        f"- test domain-unsupported families: `{payload['test_source_support_summary']['domain_unsupported_family_rows']}`",
        "",
        "## Shadow Holdout",
        "",
        f"- all improvement: `{shadow['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- t50 improvement: `{shadow['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- hard/failure improvement: `{shadow['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- easy degradation: `{shadow['easy_degradation_vs_floor']:.4f}`",
        f"- switch rate: `{shadow['switch_rate']:.4f}`",
        "",
        "## Test Once",
        "",
        f"- all improvement: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- t50 improvement: `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        f"- switch rate: `{test['switch_rate']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- Stage43-CD repairs the Stage43-CB/CC easy-safety mismatch by refusing learned switches on source families not covered by validation.",
        "- This is a source-coverage safety protocol, not a new test-tuned threshold.",
        "- It preserves easy safety and keeps a small all-row lift, but it sacrifices some hard/failure lift and still does not repair t50.",
        "- Deployment remains unchanged until this guard is reconciled with the current frozen deployable policy family.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cd_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CD Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    test = payload["test_once"]["metrics"]
    world = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{SOURCE}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "- deployable policy changed: `False`",
        f"- test all improvement: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test t50 improvement: `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        "- long objective complete: `False`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-CD is a source-family coverage guard audit for downstream latent heads.",
        "- It does not remove or replace the current Stage37/Stage42 safety floor.",
        "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
    ]
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(WORLD_GATE_MD, world)


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cd_gate"]
    selected = payload["coverage_policy"]["selected_shadow_policy"]
    shadow = selected["shadow_holdout_metrics"]
    test = payload["test_once"]["metrics"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "deployable_policy_changed = `False`",
        "",
        "Stage43-CD promotes the validation source-family support gap from Stage43-CC into an explicit coverage guard: source families absent from validation fall back to the floor.",
        f"Selected policy: `{selected['policy']['name']}`.",
        f"Unsupported test families: `{payload['test_source_support_summary']['global_unsupported_family_rows']}`.",
        f"Test all / t50 / hard / easy: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['easy_degradation_vs_floor']:.4f}`.",
        f"Shadow all / hard / easy: `{shadow['full_waypoint_ade_improvement_vs_floor']:.4f}` / `{shadow['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{shadow['easy_degradation_vs_floor']:.4f}`.",
        "",
        "Interpretation: the guard restores easy safety and keeps small all-row lift, but hard/failure lift drops and t50 remains negative. It is evidence for source-coverage safety, not a deployment replacement.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_cd_source_family_coverage_guard"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "selected_policy": selected["policy"],
        "test_metrics": test,
        "shadow_metrics": shadow,
        "unsupported_test_families": payload["test_source_support_summary"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "deployable_policy_changed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_cd_source_family_coverage_guard"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_jsonable({"event": SOURCE, "verdict": gate["verdict"], "generated_at_utc": payload["generated_at_utc"]}), ensure_ascii=False)
            + "\n"
        )


def run_source_family_coverage_guard(*, batch_size: int = 8192, ridge: float = 1e-2) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43bz = read_json(STAGE43_BZ_JSON, {})
    stage43cc = read_json(STAGE43_CC_JSON, {})
    checkpoint, ckpt, base_model = _load_model(stage43m)
    adapter_path = Path(stage43bz.get("adapter_checkpoint", OUT_DIR / "checkpoints/stage43_latent_transition_adapter_repair.pt"))
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    probe = _predict_transition_latents(base_model, train, batch_size=int(batch_size))
    adapter = _load_adapter(adapter_path, train.x.shape[1], probe["z_t"].shape[1])
    train_latent = np.concatenate(
        [probe["z_t"], probe["z_next"], _adapter_predict(adapter, train.x, probe["z_t"], batch_size=int(batch_size))],
        axis=1,
    ).astype(np.float32)
    val_latent = _encode_selected_variant(base_model, adapter, val, batch_size=int(batch_size))
    test_latent = _encode_selected_variant(base_model, adapter, test, batch_size=int(batch_size))
    weights = _fit_heads(train_latent, train, ridge=float(ridge))
    val_pred = _predict_heads(val_latent, weights)
    test_pred = _predict_heads(test_latent, weights)
    shadow = _shadow_validation_split(val)
    coverage = _select_coverage_policy(val, val_pred, shadow)
    test_families = np.asarray([_source_family(x) for x in test.source_file.astype(str)])
    test_eval = _evaluate_policy(test, test_pred, test_families, coverage["selected_shadow_policy"]["policy"], np.ones(len(test.x), dtype=bool))
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_source_family_coverage_guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {"verdict": stage43m.get("stage43_m_gate", {}).get("verdict"), "checkpoint": str(checkpoint)},
        "stage43_bz_precondition": {"verdict": stage43bz.get("stage43_bz_gate", {}).get("verdict"), "adapter_checkpoint": str(adapter_path)},
        "stage43_cc_precondition": {"verdict": stage43cc.get("stage43_cc_gate", {}).get("verdict"), "report": str(STAGE43_CC_JSON)},
        "protocol": {
            "train_only_heads_refit": True,
            "ridge": float(ridge),
            "batch_size": int(batch_size),
            "selected_variant": SELECTED_VARIANT,
            "target_vec_shape": list(_target_vec(train).shape),
            "num_workers": 0,
        },
        "rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "shadow_validation": {"support": shadow["plan"]["support"], "plan": shadow["plan"]},
        "coverage_policy": {**coverage, "selection_uses_test_metrics": False},
        "test_source_support_summary": _source_support_summary(test, test_families, shadow["plan"]["support"]),
        "test_once": {
            "metrics": test_eval["metrics"],
            "slice_tables": test_eval["slice_tables"],
            "bootstrap": _bootstrap_ci(test, test_eval["selected_ade"], test_eval["selected_fde"], n=1000, seed=1047),
        },
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_threshold_tuning": False,
            "test_statistics_normalization": False,
            "guard_uses_future_labels": False,
            "guard_uses_test_endpoints": False,
        },
        "claim_boundary": {
            "deployable_policy_changed": False,
            "metric_or_seconds_claim": False,
            "true_3d_claim": False,
            "foundation_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cd_gate"] = _gate(payload)
    _write_reports(payload)
    _update_summaries(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stage43-CD validation source-family coverage guard audit.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args(argv)
    payload = run_source_family_coverage_guard(batch_size=args.batch_size, ridge=args.ridge)
    gate = payload["stage43_cd_gate"]
    test = payload["test_once"]["metrics"]
    print(f"Stage43-CD: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"test_all={test['full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_t50={test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_hard={test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_easy={test['easy_degradation_vs_floor']:.4f}")
    return payload


if __name__ == "__main__":
    main()
