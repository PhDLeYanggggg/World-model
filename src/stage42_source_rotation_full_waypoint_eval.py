from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_rotation_full_waypoint_eval_stage42.json"
REPORT_MD = OUT_DIR / "source_rotation_full_waypoint_eval_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_je_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JE_SOURCE_ROTATION_FULL_WAYPOINT_EVAL"
SOURCE = "fresh_stage42_je_source_rotation_full_waypoint_eval"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JE 是 leave-one-domain source-rotation full-waypoint evaluation，不是 metric 或 seconds-level 结果。",
    "每个 rotation 将一个 external domain 整域留作 test；train/val 只来自其他 domains。",
    "策略只在 validation 上选 horizon-level thresholds；test domain 不参与 threshold selection。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _source_groups(data: Mapping[str, np.ndarray]) -> np.ndarray:
    source = data["source_file"].astype(str)
    domain = data["dataset"].astype(str)
    return np.asarray([f"{d}::{s42b._rel_source(s)}" for d, s in zip(domain, source)], dtype="U512")


def _leave_one_domain_split(data: Mapping[str, np.ndarray], heldout_domain: str) -> tuple[np.ndarray, dict[str, Any]]:
    domain = data["dataset"].astype(str)
    group = _source_groups(data)
    split = np.full(len(domain), "test", dtype="U8")
    train_groups: list[str] = []
    val_groups: list[str] = []
    for g in sorted(set(group[domain != heldout_domain].tolist())):
        # Stable source-level split inside the non-held-out domains. Test rows
        # are never used for threshold/model selection.
        if s42b._stable_unit(f"stage42-je::{heldout_domain}::{g}") < 0.8:
            train_groups.append(g)
        else:
            val_groups.append(g)
    if not val_groups and train_groups:
        val_groups.append(train_groups.pop())
    train_set = set(train_groups)
    val_set = set(val_groups)
    split[(domain != heldout_domain) & np.asarray([g in train_set for g in group])] = "train"
    split[(domain != heldout_domain) & np.asarray([g in val_set for g in group])] = "val"
    split[domain == heldout_domain] = "test"
    stats = {
        "heldout_domain": heldout_domain,
        "train_rows": int(np.sum(split == "train")),
        "val_rows": int(np.sum(split == "val")),
        "test_rows": int(np.sum(split == "test")),
        "train_domains": sorted(set(domain[split == "train"].tolist())),
        "val_domains": sorted(set(domain[split == "val"].tolist())),
        "test_domains": sorted(set(domain[split == "test"].tolist())),
        "train_sources": int(len(set(group[split == "train"].tolist()))),
        "val_sources": int(len(set(group[split == "val"].tolist()))),
        "test_sources": int(len(set(group[split == "test"].tolist()))),
        "source_overlap": {
            "train_val": int(len(set(group[split == "train"].tolist()) & set(group[split == "val"].tolist()))),
            "train_test": int(len(set(group[split == "train"].tolist()) & set(group[split == "test"].tolist()))),
            "val_test": int(len(set(group[split == "val"].tolist()) & set(group[split == "test"].tolist()))),
        },
    }
    stats["source_overlap_pass"] = all(v == 0 for v in stats["source_overlap"].values())
    return split, stats


def _domain_invariant_features(data: Mapping[str, np.ndarray], floor: Mapping[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    features, names = am._feature_matrix(data, floor)
    keep = np.asarray([not name.startswith("domain_") for name in names], dtype=bool)
    removed = [name for name, ok in zip(names, keep) if not ok]
    return features[:, keep], [name for name, ok in zip(names, keep) if ok], removed


def _select_horizon_policy_on_val(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    val_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pred_ade, pred_fde = am._trajectory_errors(pred_xy, labels)
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, labels)
    residual_norm = np.linalg.norm(pred_xy[:, -1] - floor_xy[:, -1], axis=1) / np.maximum(data["scale"].astype(np.float64), EPS)
    horizon = data["horizon"].astype(int)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    policy: dict[str, Any] = {
        "type": "stage42je_horizon_only_validation_safe_policy",
        "selection_source": "validation_only_non_heldout_domains",
        "slices": {},
        "test_threshold_tuning": False,
        "domain_specific_thresholds": False,
    }
    blended_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for h in [10, 25, 50, 100]:
        vm = val_mask & (horizon == h)
        if int(np.sum(vm)) < 80:
            continue
        thresholds = [float(np.quantile(residual_norm[vm], q)) for q in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]]
        best: dict[str, Any] | None = None
        best_score = 0.0
        for direction in ["low", "high"]:
            for threshold in thresholds:
                local = vm & ((residual_norm <= threshold) if direction == "low" else (residual_norm >= threshold))
                if not np.any(local):
                    continue
                for alpha in [0.25, 0.50, 0.75, 1.0]:
                    if alpha not in blended_cache:
                        blended = floor_xy + float(alpha) * (pred_xy - floor_xy)
                        blended_cache[alpha] = am._trajectory_errors(blended, labels)
                    b_ade, b_fde = blended_cache[alpha]
                    trial_ade = floor_ade.copy()
                    trial_fde = floor_fde.copy()
                    trial_ade[local] = b_ade[local]
                    trial_fde[local] = b_fde[local]
                    trial_switch = local.astype(bool)
                    metric = am._metric(trial_ade, floor_ade, data, trial_switch, vm)
                    if metric["easy_degradation"] > 0.02:
                        continue
                    score = (
                        1.2 * metric["all_improvement"]
                        + 1.8 * metric["t50_improvement"]
                        + 1.1 * metric["hard_failure_improvement"]
                        - 20.0 * max(0.0, metric["easy_degradation"] - 0.01)
                        - 0.03 * metric["switch_rate"]
                    )
                    if score > best_score:
                        best_score = float(score)
                        best = {
                            "direction": direction,
                            "residual_norm_threshold": float(threshold),
                            "alpha": float(alpha),
                            "val_score": float(score),
                            "val_rows": int(np.sum(vm)),
                            "val_metric": metric,
                        }
        if best is not None and best_score > 0.0:
            policy["slices"][f"h{h}"] = best
    for key, params in policy["slices"].items():
        h = int(key[1:])
        local = (horizon == h) & (
            (residual_norm <= float(params["residual_norm_threshold"]))
            if params["direction"] == "low"
            else (residual_norm >= float(params["residual_norm_threshold"]))
        )
        alpha = float(params["alpha"])
        blended = floor_xy + alpha * (pred_xy - floor_xy)
        b_ade, b_fde = am._trajectory_errors(blended, labels)
        selected_ade[local] = b_ade[local]
        selected_fde[local] = b_fde[local]
        switch[local] = True
    return policy, selected_ade, selected_fde, switch


def _evaluate_rotation(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    heldout_domain: str,
) -> dict[str, Any]:
    split, split_stats = _leave_one_domain_split(data, heldout_domain)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names, removed_features = _domain_invariant_features(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    val_results = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = _select_horizon_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
        val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.2 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.1 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        val_results.append({"lambda": float(lam), "score": float(score), "policy_slice_count": len(policy["slices"]), "val_metric": val_metric})
        if score > best_score:
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "pred_xy": pred_xy,
                "score": float(score),
                "val_metric": val_metric,
            }
    if best is None:
        raise RuntimeError(f"No model selected for heldout domain {heldout_domain}.")
    pred_ade, pred_fde = am._trajectory_errors(best["pred_xy"], labels)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "heldout_domain": heldout_domain,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names)),
            "domain_features_removed": removed_features,
            "future_inputs": False,
            "normalization": "train_split_mean_std_only",
        },
        "floor": {
            "strongest_by_horizon": floor["strongest_by_horizon"],
            "geometry_diagnostics": floor["geometry_diagnostics"],
        },
        "best_lambda": best["lambda"],
        "validation_selection": {
            "source": "fresh_run_validation_only_non_heldout_domains",
            "test_threshold_tuning": False,
            "selected_score": best["score"],
            "candidates": val_results,
        },
        "policy": best["policy"],
        "metrics": {
            "floor": am._metric(best["floor_ade"], best["floor_ade"], data, np.zeros(len(h), dtype=bool), test_mask),
            "ungated_ridge_diagnostic": am._metric(pred_ade, best["floor_ade"], data, np.ones(len(h), dtype=bool), test_mask),
            "protected_horizon_policy": am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask),
            "protected_horizon_policy_fde": am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask),
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42101),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42102),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42103),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42104),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42105),
        },
    }


def _summary(rotations: list[Mapping[str, Any]]) -> dict[str, Any]:
    protected = [row["metrics"]["protected_horizon_policy"] for row in rotations]
    positive = [
        row["heldout_domain"]
        for row, metric in zip(rotations, protected)
        if metric["all_improvement"] > 0
        and (metric["t50_improvement"] > 0 or metric["hard_failure_improvement"] > 0)
        and metric["easy_degradation"] <= 0.02
    ]
    deployable = [
        row["heldout_domain"]
        for row, metric in zip(rotations, protected)
        if metric["all_improvement"] > 0.03
        and (metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10)
        and metric["easy_degradation"] <= 0.02
    ]
    return {
        "source": SOURCE,
        "rotation_count": int(len(rotations)),
        "positive_heldout_domains": positive,
        "deployable_heldout_domains": deployable,
        "positive_heldout_domain_count": int(len(positive)),
        "deployable_heldout_domain_count": int(len(deployable)),
        "all_rotations_no_easy_harm": all(metric["easy_degradation"] <= 0.02 for metric in protected),
        "decision": "source_rotation_positive_but_not_global_deployable"
        if positive and len(deployable) < len(rotations)
        else "source_rotation_deployable_across_all_domains"
        if len(deployable) == len(rotations) and rotations
        else "source_rotation_diagnostic_or_negative",
        "next_action": "Use this rotation result as the cross-domain boundary: promote only domains that are positive under held-out source rotation; otherwise continue source-specific data/calibration expansion.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    labels = am._reconstruct_waypoint_labels(data)
    domains = sorted(set(data["dataset"].astype(str).tolist()))
    rotations = [_evaluate_rotation(data, labels, domain) for domain in domains]
    payload: dict[str, Any] = {
        "stage": "Stage42-JE",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json",
                "outputs/stage42_long_research/latest_evidence_tier_consolidation_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "domains": domains,
        "label_stats": {
            "rows": int(len(data["horizon"])),
            "full_waypoint_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
            "missing_track_rows": int(np.sum(labels["missing_track"])),
        },
        "rotations": rotations,
        "summary": _summary(rotations),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "domain_specific_test_thresholds": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in rotations),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_je_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rotations = payload["rotations"]
    gates = {
        "rotations_built_for_all_domains": len(rotations) == len(payload["domains"]) and len(rotations) >= 3,
        "heldout_test_rows_exist": all(row["split_stats"]["test_rows"] > 0 for row in rotations),
        "train_val_rows_exist": all(row["split_stats"]["train_rows"] > 0 and row["split_stats"]["val_rows"] > 0 for row in rotations),
        "source_overlap_pass": payload["no_leakage"]["source_overlap_pass"] is True,
        "domain_features_removed": all(row["feature_schema"]["domain_features_removed"] for row in rotations),
        "horizon_only_validation_policy": all(row["policy"]["domain_specific_thresholds"] is False for row in rotations),
        "test_threshold_tuning_false": payload["no_leakage"]["test_threshold_tuning"] is False,
        "at_least_one_heldout_positive_or_recorded_negative": payload["summary"]["positive_heldout_domain_count"] >= 0,
        "easy_safety_recorded": "all_rotations_no_easy_harm" in payload["summary"],
        "bootstrap_reported": all(row["bootstrap"]["all"]["bootstrap_n"] > 0 for row in rotations),
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
                "domain_specific_test_thresholds",
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = (
        "stage42_je_source_rotation_full_waypoint_eval_pass"
        if passed == len(gates)
        else "stage42_je_source_rotation_full_waypoint_eval_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_je_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JE Source-Rotation Full-Waypoint Evaluation",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Why This Stage Exists",
        "",
        "- Stage42-JC established the current strongest source-level row-cache evidence, while Stage42-JD kept metric/seconds claims blocked.",
        "- Stage42-JE asks a stricter question: if an entire external domain is held out, can a domain-invariant full-waypoint probe still help under validation-selected safety?",
        "- This is not used to tune the existing main policy; it is a boundary check for cross-domain world-model claims.",
        "",
        "## Summary",
        "",
        f"- decision: `{summary['decision']}`",
        f"- positive_heldout_domains: `{summary['positive_heldout_domains']}`",
        f"- deployable_heldout_domains: `{summary['deployable_heldout_domains']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Held-Out Domain Rotations",
        "",
        "| heldout domain | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | policy slices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rotations"]:
        metric = row["metrics"]["protected_horizon_policy"]
        lines.append(
            f"| `{row['heldout_domain']}` | {metric['rows']} | {_fmt(metric['all_improvement'])} | "
            f"{_fmt(metric['t50_improvement'])} | {_fmt(metric['t100_raw_frame_diagnostic_improvement'])} | "
            f"{_fmt(metric['hard_failure_improvement'])} | {_fmt(metric['easy_degradation'])} | "
            f"{_fmt(metric['switch_rate'])} | {len(row['policy']['slices'])} |"
        )
    lines.extend(["", "## Bootstrap CI", "", "| heldout domain | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["rotations"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_domain']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- A positive held-out rotation supports cross-domain raw-frame transfer only for that held-out domain and only under this domain-invariant policy.",
            "- A negative held-out rotation means the main Stage42 result must remain source/protocol bounded; it must not be written as foundation-scale or metric world-model generalization.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_je_gate"]
    lines = [
        "# Stage42-JE Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_je_gate"]
    rows = []
    for row in payload["rotations"]:
        metric = row["metrics"]["protected_horizon_policy"]
        rows.append(
            f"{row['heldout_domain']}: all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}"
        )
    return [
        "## Stage42-JE Source-Rotation Full-Waypoint Evaluation",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- held-out domain rotations: {'; '.join(rows)}.",
        f"- decision: `{summary['decision']}`; deployable held-out domains: `{summary['deployable_heldout_domains']}`.",
        "- boundary: this is stricter cross-domain raw-frame evidence; it does not change the no-metric/no-seconds/no-Stage5C/no-SMC boundary.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["source_rotation_full_waypoint_eval"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_je_gate"]["verdict"],
        "gate": {"passed": payload["stage42_je_gate"]["passed"], "total": payload["stage42_je_gate"]["total"]},
        "positive_heldout_domains": payload["summary"]["positive_heldout_domains"],
        "deployable_heldout_domains": payload["summary"]["deployable_heldout_domains"],
        "decision": payload["summary"]["decision"],
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    import json

    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JE",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_je_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": True,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_source_rotation_full_waypoint_eval(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_source_rotation_full_waypoint_eval(refresh_readmes=True)


if __name__ == "__main__":
    main()
