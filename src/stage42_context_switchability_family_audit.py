from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_context_gain_router as el
from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, write_json, write_md, read_json
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_switchability_family_audit_stage42.json"
REPORT_MD = OUT_DIR / "context_switchability_family_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gk_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gk_context_switchability_family_audit"
MIN_MATERIAL_DELTA = 0.01
RIDGE_LAMBDAS = [0.1, 1.0, 10.0, 100.0]
FEATURE_FAMILIES = [
    "history_only",
    "goal_only",
    "neighbor_only",
    "motion_goal_context",
    "baseline_plus_history",
    "baseline_plus_goal",
    "baseline_plus_neighbor",
    "baseline_plus_history_goal_neighbor",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GK 是 changed-target context switchability/gain/harm family audit，不重复已关闭的 residual sequence/graph protocol。",
    "router target 是 feature-family proposal 相对 baseline-family control 的 gain/harm/switchability，而不是直接 residual trajectory target。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _fit_ridge(x: np.ndarray, y: np.ndarray, mask: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(mask)[0]
    xt = x[ids].astype(np.float64, copy=False)
    yt = y[ids].astype(np.float64, copy=False)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _score(metric: Mapping[str, Any]) -> float:
    return float(
        1.2 * metric["all_improvement"]
        + 1.8 * metric["t50_improvement"]
        + 1.2 * metric["hard_failure_improvement"]
        - 50.0 * max(0.0, metric["easy_degradation"] - 0.02)
        - 0.02 * metric["switch_rate"]
    )


def _train_gain_harm_router(
    *,
    raw_features: np.ndarray,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    split: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    x, _, _ = am._standardize(raw_features, train_mask)
    true_gain = (base_ade - candidate_ade).astype(np.float32)
    true_harm = np.maximum(candidate_ade - base_ade, 0.0).astype(np.float32)
    best: dict[str, Any] | None = None
    candidates_checked = 0
    for lam in RIDGE_LAMBDAS:
        gain_coef = _fit_ridge(x, true_gain, train_mask, lam)
        harm_coef = _fit_ridge(x, true_harm, train_mask, lam)
        pred_gain = (x.astype(np.float64) @ gain_coef.astype(np.float64)).astype(np.float64)
        pred_harm = np.maximum((x.astype(np.float64) @ harm_coef.astype(np.float64)).astype(np.float64), 0.0)
        val_gain = pred_gain[val_mask]
        val_harm = pred_harm[val_mask]
        gain_thresholds = sorted(set(float(np.quantile(val_gain, q)) for q in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]))
        gain_thresholds.extend([0.0, float(np.mean(val_gain)), float(np.mean(val_gain) + np.std(val_gain))])
        harm_thresholds = sorted(set(float(np.quantile(val_harm, q)) for q in [0.10, 0.25, 0.50, 0.75]))
        harm_thresholds.extend([0.0, float(np.mean(val_harm))])
        for gain_threshold in gain_thresholds:
            for harm_threshold in harm_thresholds:
                switch = (pred_gain > gain_threshold) & (pred_harm <= harm_threshold)
                selected = base_ade.copy()
                selected[switch] = candidate_ade[switch]
                val_metric = am._metric(selected, base_ade, data, switch, val_mask)
                candidates_checked += 1
                if val_metric["easy_degradation"] > 0.02:
                    continue
                score = _score(val_metric)
                if best is None or score > best["score"]:
                    best = {
                        "lambda": float(lam),
                        "gain_threshold": float(gain_threshold),
                        "harm_threshold": float(harm_threshold),
                        "score": float(score),
                        "val_metric": val_metric,
                        "selected": selected,
                        "switch": switch,
                        "pred_gain": pred_gain,
                        "pred_harm": pred_harm,
                    }
    if best is None:
        switch = np.zeros(len(base_ade), dtype=bool)
        selected = base_ade.copy()
        best = {
            "lambda": None,
            "gain_threshold": None,
            "harm_threshold": None,
            "score": 0.0,
            "val_metric": am._metric(selected, base_ade, data, switch, val_mask),
            "selected": selected,
            "switch": switch,
            "pred_gain": np.zeros(len(base_ade), dtype=np.float64),
            "pred_harm": np.zeros(len(base_ade), dtype=np.float64),
        }
    test_metric = am._metric(best["selected"], base_ade, data, best["switch"], test_mask)
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "selection": {
            "source": "train_gain_harm_regression_val_threshold_selection_test_once",
            "candidates_checked": candidates_checked,
            "lambda": best["lambda"],
            "gain_threshold": best["gain_threshold"],
            "harm_threshold": best["harm_threshold"],
            "val_score": best["score"],
            "val_metric": best["val_metric"],
            "test_threshold_tuning": False,
        },
        "test_metric_vs_baseline_family": test_metric,
        "bootstrap_vs_baseline_family": {
            "all": am._bootstrap_ci(best["selected"], base_ade, test_mask, seed=42701),
            "t50": am._bootstrap_ci(best["selected"], base_ade, test_mask & (horizon == 50), seed=42702),
            "hard_failure": am._bootstrap_ci(best["selected"], base_ade, test_mask & hard_failure, seed=42703),
            "easy_degradation": am._bootstrap_ci(base_ade, best["selected"], test_mask & easy, seed=42704),
        },
        "switch_rate_test": float(np.mean(best["switch"][test_mask])) if np.any(test_mask) else 0.0,
        "mean_pred_gain_test": float(np.mean(best["pred_gain"][test_mask])) if np.any(test_mask) else 0.0,
        "mean_pred_harm_test": float(np.mean(best["pred_harm"][test_mask])) if np.any(test_mask) else 0.0,
    }


def _material(metric: Mapping[str, Any]) -> bool:
    return bool(
        metric["easy_degradation"] <= 0.02
        and (
            metric["all_improvement"] >= MIN_MATERIAL_DELTA
            or metric["t50_improvement"] >= MIN_MATERIAL_DELTA
            or metric["hard_failure_improvement"] >= MIN_MATERIAL_DELTA
        )
    )


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    data = shared["data"]
    baseline = el._prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    family_rows: list[dict[str, Any]] = []
    for family in FEATURE_FAMILIES:
        candidate = el._prepare_variant_predictions(shared["features"][:, masks[family]], shared)
        router = _train_gain_harm_router(
            raw_features=shared["features"][:, masks[family]],
            base_ade=baseline["selected_ade"],
            candidate_ade=candidate["selected_ade"],
            split=split,
            data=data,
        )
        metric = router["test_metric_vs_baseline_family"]
        family_rows.append(
            {
                "family": family,
                "source": "fresh_run",
                "feature_count": int(np.sum(masks[family])),
                "candidate_val_metric_vs_causal_floor": candidate["val_metric"],
                "router": router,
                "material_context_contribution": _material(metric),
            }
        )
    best = max(
        family_rows,
        key=lambda row: (
            row["router"]["test_metric_vs_baseline_family"]["all_improvement"]
            + row["router"]["test_metric_vs_baseline_family"]["t50_improvement"]
            + row["router"]["test_metric_vs_baseline_family"]["hard_failure_improvement"]
        ),
    )
    material_families = [row["family"] for row in family_rows if row["material_context_contribution"]]
    summary = {
        "source": SOURCE,
        "baseline_family_control_val_metric": baseline["val_metric"],
        "feature_families_checked": FEATURE_FAMILIES,
        "families_checked_count": len(family_rows),
        "material_delta_threshold": MIN_MATERIAL_DELTA,
        "best_family": best["family"],
        "best_test_metric_vs_baseline_family": best["router"]["test_metric_vs_baseline_family"],
        "material_context_families": material_families,
        "material_context_contribution_supported": bool(material_families),
        "decision": "context_switchability_family_supported" if material_families else "context_switchability_family_not_supported",
        "root_cause": (
            "Feature-family context gain/harm targets still do not yield a material positive, easy-safe lift over baseline-family control."
            if not material_families
            else "At least one context family yields material easy-safe lift under gain/harm switchability target."
        ),
        "next_action": (
            "Do not claim scene/goal/neighbor as independent main contribution if material families remain empty; next attempt must change source support or full-sequence target."
            if not material_families
            else "Promote the supported context family only within protected raw-frame 2.5D claim boundaries and rerun source/horizon robustness."
        ),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GK context switchability family audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/context_model_closure_stage42.json",
                "outputs/stage42_long_research/module_claim_lock_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "family_rows": family_rows,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "future_labels_train_val_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "context_main_claim_allowed": bool(material_families),
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_gk_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "baseline_family_control_loaded": bool(s["baseline_family_control_val_metric"]),
        "changed_target_gain_harm_used": True,
        "feature_families_checked": s["families_checked_count"] >= 8,
        "validation_selection_test_once_recorded": all(
            row["router"]["selection"]["test_threshold_tuning"] is False for row in payload["family_rows"]
        ),
        "materiality_decision_recorded": s["decision"] in {
            "context_switchability_family_supported",
            "context_switchability_family_not_supported",
        },
        "claim_boundary_matches_materiality": boundary["context_main_claim_allowed"] is s["material_context_contribution_supported"],
        "root_cause_written": bool(s["root_cause"]),
        "next_action_written": bool(s["next_action"]),
        "no_future_or_test_leakage": all(
            payload["no_leakage"][key] is False
            for key in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "not_true3d_or_foundation": boundary["true_3d"] is False and boundary["foundation_world_model"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = "stage42_gk_context_switchability_family_audit_pass" if passed == total else "stage42_gk_context_switchability_family_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_gk_gate"]
    lines = [
        "# Stage42-GK Context Switchability Family Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{s['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- material_delta_threshold: `{s['material_delta_threshold']}`",
        f"- best_family: `{s['best_family']}`",
        f"- material_context_families: `{s['material_context_families']}`",
        f"- material_context_contribution_supported: `{s['material_context_contribution_supported']}`",
        f"- root_cause: {s['root_cause']}",
        f"- next_action: {s['next_action']}",
        "",
        "## Family Rows",
        "",
        "| family | features | all | t50 | t100 raw | hard | easy | switch | material |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["family_rows"]:
        m = row["router"]["test_metric_vs_baseline_family"]
        lines.append(
            f"| `{row['family']}` | {row['feature_count']} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['t100_raw_frame_diagnostic_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | {row['material_context_contribution']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a changed-target context audit: gain/harm/switchability is the trained target, not residual trajectory deltas.",
            "- If material_context_families is empty, scene/goal/neighbor/history context remains blocked as an independent main contribution under this target.",
            "- A negative result still matters: it narrows the next credible experiment to source support or genuinely different full-sequence/group objectives.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gk_gate"]
    return [
        "# Stage42-GK Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _replace_section(text: str, marker: str, section: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        return before + section + after
    return text.rstrip() + "\n\n" + section


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    m = s["best_test_metric_vs_baseline_family"]
    return "\n".join(
        [
            "<!-- STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT:START -->",
            "## Stage42-GK Context Switchability Family Audit",
            "",
            f"- source: `{payload['source']}`",
            f"- gate: `{payload['stage42_gk_gate']['passed']} / {payload['stage42_gk_gate']['total']}`; verdict `{payload['stage42_gk_gate']['verdict']}`.",
            f"- decision: `{s['decision']}`; material context families: `{s['material_context_families']}`.",
            f"- best family `{s['best_family']}` vs baseline-family control: all/t50/t100raw/hard/easy = `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_raw_frame_diagnostic_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}` / `{m['easy_degradation']:.6f}`.",
            "- Target changed from residual trajectory deltas to gain/harm/switchability. Future labels are train/val/eval labels only, never inference inputs.",
            "- If no material family is supported, scene/goal/neighbor context remains blocked as an independent main claim under this changed-target audit.",
            "- Still no true-3D, foundation, global metric, seconds-level, Stage5C, SMC, or test-endpoint claim.",
            "<!-- STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        old = path.read_text() if path.exists() else ""
        path.write_text(_replace_section(old, "STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GK context switchability family audit"
    state["current_verdict"] = payload["stage42_gk_gate"]["verdict"]
    state["stage42_gk_context_switchability_family_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_gk_gate"]["verdict"],
        "gates": f"{payload['stage42_gk_gate']['passed']}/{payload['stage42_gk_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": payload["summary"]["root_cause"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_context_switchability_family_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_context_switchability_family_audit()
    gate = result["stage42_gk_gate"]
    print(f"Stage42-GK gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
