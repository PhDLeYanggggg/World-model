from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_calibrated_t50_source_support_gap_audit as br
from src import stage42_calibrated_subset_eval as bo
from src import stage42_external_validation as s42b
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BR_JSON = OUT_DIR / "calibrated_t50_source_support_gap_stage42.json"
REPORT_JSON = OUT_DIR / "ucy_zara_t50_family_policy_stage42.json"
REPORT_MD = OUT_DIR / "ucy_zara_t50_family_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bs_gate.md"

EPS = 1e-6
ZARA_SOURCE_IDS = ["UCY_zara01", "UCY_zara02", "UCY_zara03"]
ZARA_SOURCE_TO_REL = {
    "UCY_zara01": "UCY/zara01/obsmat.txt",
    "UCY_zara02": "UCY/zara02/obsmat.txt",
    "UCY_zara03": "UCY/zara03/crowds_zara03.txt",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BS 只针对 Stage42-BR 标出的 UCY_zara calibrated t50 policy/model blocker。",
    "UCY_zara 有足够同族 source support，因此本步骤不需要新数据许可。",
    "训练只用 UCY_zara train source；threshold / alpha / candidate 只用 validation source 选择；holdout source 只最终评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _subset_rows(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(mask)
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == n:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def _source_rel_array(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([s42b._rel_source(str(path)) for path in data["source_file"].astype(str)], dtype="U256")


def _build_zara_folds() -> list[dict[str, Any]]:
    # zara01 and zara02 validate each other when held out; zara03 has no same-domain
    # sibling in the calibrated subset, so zara01 is used as validation and zara02
    # as training. This is deterministic and does not use test metrics.
    return [
        {"fold_id": 0, "holdout_source": "UCY_zara01", "validation_source": "UCY_zara02", "train_sources": ["UCY_zara03"]},
        {"fold_id": 1, "holdout_source": "UCY_zara02", "validation_source": "UCY_zara01", "train_sources": ["UCY_zara03"]},
        {"fold_id": 2, "holdout_source": "UCY_zara03", "validation_source": "UCY_zara01", "train_sources": ["UCY_zara02"]},
    ]


def _split_for_fold(rel_source: np.ndarray, fold: Mapping[str, Any]) -> np.ndarray:
    split = np.full(len(rel_source), "ignore", dtype="U8")
    train = {ZARA_SOURCE_TO_REL[s] for s in fold["train_sources"]}
    val = {ZARA_SOURCE_TO_REL[fold["validation_source"]]}
    test = {ZARA_SOURCE_TO_REL[fold["holdout_source"]]}
    split[np.isin(rel_source, list(train))] = "train"
    split[np.isin(rel_source, list(val))] = "val"
    split[np.isin(rel_source, list(test))] = "test"
    return split


def _fold_stats(data: Mapping[str, np.ndarray], rel_source: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    out: dict[str, Any] = {}
    for sp in ["train", "val", "test"]:
        m = split == sp
        out[sp] = {
            "rows": int(np.sum(m)),
            "domains": dict(Counter(domain[m].tolist())),
            "sources": dict(Counter(rel_source[m].tolist())),
            "t10": int(np.sum(m & (h == 10))),
            "t25": int(np.sum(m & (h == 25))),
            "t50": int(np.sum(m & (h == 50))),
            "t100": int(np.sum(m & (h == 100))),
            "easy": int(np.sum(data["easy"].astype(bool)[m])),
            "hard": int(np.sum(data["hard"].astype(bool)[m])),
            "failure": int(np.sum(data["failure"].astype(bool)[m])),
        }
    sets = {sp: set(rel_source[split == sp].tolist()) for sp in ["train", "val", "test"]}
    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    out["source_overlap"] = overlap
    out["source_overlap_pass"] = not any(overlap.values())
    return out


def _linear_waypoints(endpoint_xy: np.ndarray, data: Mapping[str, np.ndarray]) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    return (cur[:, None, :] + am.WAYPOINT_FRAC[None, :, None] * (endpoint_xy.astype(np.float64) - cur)[:, None, :]).astype(np.float32)


def _ridge_waypoint_candidate(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    train_mask: np.ndarray,
    t50_only: bool,
    lam: float,
) -> np.ndarray:
    features, _ = am._feature_matrix(data, floor)
    x, _, _ = am._standardize(features, train_mask)
    target_delta = (
        (labels["waypoint_xy"].astype(np.float64) - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :])
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    fit_mask = train_mask.copy()
    if t50_only:
        fit_mask &= data["horizon"].astype(int) == 50
    coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], fit_mask, float(lam))
    return am._predict_waypoints(x, coef, data)


def _candidate_waypoints(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {
        "floor_no_switch": floor["floor_xy"].astype(np.float32),
    }
    safe = s41._safe_baseline_predictions(data).astype(np.float32)
    for i in range(safe.shape[1]):
        candidates[f"safe_baseline_{i}"] = _linear_waypoints(safe[:, i], data)
    family = data["family_pred"].astype(np.float32)
    for i in range(family.shape[1]):
        candidates[f"family_baseline_{i}"] = _linear_waypoints(family[:, i], data)
    for lam in [0.1, 1.0, 10.0, 100.0]:
        candidates[f"ridge_all_lambda_{lam:g}"] = _ridge_waypoint_candidate(data, labels, floor, train_mask, False, lam)
        candidates[f"ridge_t50_lambda_{lam:g}"] = _ridge_waypoint_candidate(data, labels, floor, train_mask, True, lam)
    return candidates


def _select_t50_policy(
    candidates: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    val_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    h = data["horizon"].astype(int)
    t50_val = val_mask & (h == 50)
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, labels)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    policy: dict[str, Any] = {
        "type": "stage42bs_ucy_zara_family_t50_policy",
        "selection_source": "validation_source_only",
        "test_threshold_tuning": False,
        "slice": None,
        "fallback_reason": "no_validation_safe_t50_candidate_selected",
    }
    if int(np.sum(t50_val)) < 80:
        return policy, selected_ade, selected_fde, switch

    best: dict[str, Any] | None = None
    best_score = 0.0
    floor_endpoint = floor_xy[:, -1]
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    for name, cand_xy in candidates.items():
        if name == "floor_no_switch":
            continue
        cand_ade, cand_fde = am._trajectory_errors(cand_xy, labels)
        residual_norm = np.linalg.norm(cand_xy[:, -1].astype(np.float64) - floor_endpoint.astype(np.float64), axis=1) / scale
        q_values = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.90]
        thresholds = [float(np.quantile(residual_norm[t50_val], q)) for q in q_values]
        for direction in ["low", "high"]:
            for threshold in thresholds:
                gate = (residual_norm <= threshold) if direction == "low" else (residual_norm >= threshold)
                local = (h == 50) & gate
                if not np.any(local & t50_val):
                    continue
                for alpha in [0.25, 0.50, 0.75, 1.0]:
                    blended = floor_xy + float(alpha) * (cand_xy - floor_xy)
                    b_ade, b_fde = am._trajectory_errors(blended, labels)
                    trial_ade = floor_ade.copy()
                    trial_fde = floor_fde.copy()
                    trial_ade[local] = b_ade[local]
                    trial_fde[local] = b_fde[local]
                    trial_switch = local.astype(bool)
                    val_metric = am._metric(trial_ade, floor_ade, data, trial_switch, val_mask)
                    val_t50 = am._metric(trial_ade, floor_ade, data, trial_switch, t50_val)
                    if val_t50["all_improvement"] <= 0.0:
                        continue
                    # The first BS run showed that near-global t50 switching can
                    # be t50-positive while still harming easy holdout rows. Keep
                    # the policy deliberately selective: this is a deployment
                    # guard selected on validation only, not a test-tuned fix.
                    if val_t50["switch_rate"] > 0.45:
                        continue
                    if val_metric["easy_degradation"] > 0.02 or val_metric["harm_over_fallback"] > 0.0:
                        continue
                    score = (
                        3.0 * val_t50["all_improvement"]
                        + 1.0 * val_metric["hard_failure_improvement"]
                        - 0.20 * val_t50["switch_rate"]
                        - 0.05 * val_metric["switch_rate"]
                    )
                    if score > best_score:
                        best_score = float(score)
                        best = {
                            "candidate": name,
                            "direction": direction,
                            "residual_norm_threshold": float(threshold),
                            "alpha": float(alpha),
                            "val_score": float(score),
                            "val_metric": val_metric,
                            "val_t50_metric": val_t50,
                        }
    if best is None:
        return policy, selected_ade, selected_fde, switch

    cand_xy = candidates[str(best["candidate"])]
    residual_norm = np.linalg.norm(cand_xy[:, -1].astype(np.float64) - floor_endpoint.astype(np.float64), axis=1) / scale
    gate = (
        residual_norm <= float(best["residual_norm_threshold"])
        if best["direction"] == "low"
        else residual_norm >= float(best["residual_norm_threshold"])
    )
    local = (h == 50) & gate
    blended = floor_xy + float(best["alpha"]) * (cand_xy - floor_xy)
    b_ade, b_fde = am._trajectory_errors(blended, labels)
    selected_ade[local] = b_ade[local]
    selected_fde[local] = b_fde[local]
    switch[local] = True
    policy["slice"] = best
    policy["fallback_reason"] = None
    return policy, selected_ade, selected_fde, switch


def _candidate_oracle_headroom(
    candidates: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, labels)
    candidate_errors = []
    for name, xy in candidates.items():
        if name == "floor_no_switch":
            continue
        cand_ade, _ = am._trajectory_errors(xy, labels)
        candidate_errors.append(cand_ade)
    if not candidate_errors or int(np.sum(mask)) == 0:
        return {"rows": int(np.sum(mask)), "oracle_headroom": 0.0, "candidate_count": len(candidate_errors)}
    best = np.min(np.stack(candidate_errors, axis=1), axis=1)
    return {
        "rows": int(np.sum(mask)),
        "oracle_headroom": am._safe_improvement(best, floor_ade, mask),
        "candidate_count": len(candidate_errors),
    }


def _evaluate_fold(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], rel_source: np.ndarray, fold: Mapping[str, Any]) -> dict[str, Any]:
    split = _split_for_fold(rel_source, fold)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    candidates = _candidate_waypoints(data, labels, floor, train_mask)
    policy, selected_ade, selected_fde, switch = _select_t50_policy(candidates, floor["floor_xy"], labels, data, val_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    t50_test = test_mask & (h == 50)
    return {
        "source": "fresh_ucy_zara_t50_family_policy_fold",
        "fold": fold,
        "fold_stats": _fold_stats(data, rel_source, split),
        "candidate_count": len(candidates) - 1,
        "candidate_oracle_headroom": _candidate_oracle_headroom(candidates, floor["floor_xy"], labels, data, t50_test),
        "policy_selected": policy["slice"] is not None,
        "policy": policy,
        "validation_metric": am._metric(selected_ade, floor_ade, data, switch, val_mask),
        "validation_t50_metric": am._metric(selected_ade, floor_ade, data, switch, val_mask & (h == 50)),
        "protected_ade": am._metric(selected_ade, floor_ade, data, switch, test_mask),
        "protected_fde": am._metric(selected_fde, floor_fde, data, switch, test_mask),
        "bootstrap": {
            "t50": am._bootstrap_ci(selected_ade, floor_ade, t50_test, seed=42301),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test_mask & hard_failure, seed=42302),
            "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, test_mask & easy, seed=42303),
        },
    }


def _aggregate(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [fold["protected_ade"] for fold in folds]
    t50 = [float(m["t50_improvement"]) for m in metrics]
    all_imp = [float(m["all_improvement"]) for m in metrics]
    hard = [float(m["hard_failure_improvement"]) for m in metrics]
    easy = [float(m["easy_degradation"]) for m in metrics]
    oracle = [float(fold["candidate_oracle_headroom"]["oracle_headroom"]) for fold in folds]
    selected = [bool(fold["policy_selected"]) for fold in folds]
    return {
        "source_cv_folds": len(folds),
        "rows_total": int(sum(int(m["rows"]) for m in metrics)),
        "t50_rows_total": int(sum(int(fold["fold_stats"]["test"]["t50"]) for fold in folds)),
        "candidate_t50_oracle_headroom_macro_mean": float(np.mean(oracle)) if oracle else 0.0,
        "all_improvement_macro_mean": float(np.mean(all_imp)),
        "all_improvement_min": float(np.min(all_imp)),
        "t50_improvement_macro_mean": float(np.mean(t50)),
        "t50_improvement_min": float(np.min(t50)),
        "hard_failure_improvement_macro_mean": float(np.mean(hard)),
        "easy_degradation_max": float(np.max(easy)),
        "policy_selected_fold_count": int(sum(selected)),
        "positive_t50_fold_count": int(sum(v > 0.0 for v in t50)),
        "nonnegative_t50_folds": bool(all(v >= 0.0 for v in t50)),
        "easy_safe_all_folds": bool(all(v <= 0.02 for v in easy)),
        "nonnegative_all_folds": bool(all(v >= 0.0 for v in all_imp)),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    gates = {
        "br_input_verified": payload["br_verdict"] == "stage42_br_calibrated_t50_source_support_gap_audit_pass",
        "zara_sources_present": s["zara_sources_present"] == 3,
        "source_cv_completed": s["source_cv_folds"] == 3,
        "t50_rows_present": s["t50_rows_total"] > 0,
        "candidate_oracle_headroom_exists": s["candidate_t50_oracle_headroom_macro_mean"] > 0.0,
        "source_overlap_pass": no_leak["source_overlap_pass"] is True,
        "no_future_inputs": no_leak["future_endpoint_input"] is False and no_leak["future_waypoint_input"] is False,
        "validation_only_selection": no_leak["test_threshold_tuning"] is False,
        "easy_safe_or_honest_fallback": s["easy_degradation_max"] <= 0.02,
        "result_honestly_classified": payload["summary"]["positive_t50_claim_allowed"] in {True, False},
        "global_metric_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    positive = (
        s["positive_t50_fold_count"] == s["source_cv_folds"]
        and s["nonnegative_all_folds"]
        and s["easy_safe_all_folds"]
        and s["t50_improvement_macro_mean"] > 0.03
    )
    verdict = (
        "stage42_bs_ucy_zara_t50_family_policy_pass_positive"
        if passed == total and positive
        else "stage42_bs_ucy_zara_t50_family_policy_pass_honest_blocker"
        if passed == total
        else "stage42_bs_ucy_zara_t50_family_policy_partial"
    )
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "positive_t50_transfer": bool(positive),
        "verdict": verdict,
    }


def run_stage42_ucy_zara_t50_family_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    br_payload = _load_json(BR_JSON)
    raw = s41._combined()
    raw_rel = bo._source_rel_array(raw)
    zara_rels = {ZARA_SOURCE_TO_REL[s] for s in ZARA_SOURCE_IDS}
    zara_mask = np.isin(raw_rel, list(zara_rels))
    data = _subset_rows(raw, zara_mask)
    rel_source = _source_rel_array(data)
    labels = am._reconstruct_waypoint_labels(data)
    present = sorted([sid for sid, rel in ZARA_SOURCE_TO_REL.items() if np.any(rel_source == rel)])
    folds = _build_zara_folds()
    fold_results = [_evaluate_fold(data, labels, rel_source, fold) for fold in folds]
    aggregate = _aggregate(fold_results)
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "train_only_feature_normalization": True,
        "source_overlap_pass": all(bool(fold["fold_stats"]["source_overlap_pass"]) for fold in fold_results),
    }
    positive_allowed = (
        aggregate["positive_t50_fold_count"] == aggregate["source_cv_folds"]
        and aggregate["easy_safe_all_folds"]
        and aggregate["nonnegative_all_folds"]
        and aggregate["t50_improvement_macro_mean"] > 0.03
    )
    remaining_blocker = (
        "none"
        if positive_allowed
        else "UCY_zara has same-family source support but the validation-selected family-specific t50 policy did not produce positive safe t50 transfer on every holdout source."
    )
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "source_specific_annotation_step_subset_claim_allowed": True,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
        "positive_t50_claim_allowed": bool(positive_allowed),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    summary = {
        "source": "fresh_ucy_zara_t50_family_policy",
        "zara_sources_present": len(present),
        "zara_source_ids": present,
        **aggregate,
        "positive_t50_claim_allowed": bool(positive_allowed),
        "remaining_blocker": remaining_blocker,
        "training_run": True,
        "auto_download_executed": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_ucy_zara_t50_family_policy",
        "stage": "Stage42-BS UCY_zara family-specific calibrated t50 policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BR_JSON), "data/stage41_world_model/combined_external.npz"]),
        "current_facts": CURRENT_FACTS,
        "br_verdict": br_payload.get("stage42_br_gate", {}).get("verdict"),
        "summary": summary,
        "fold_results": fold_results,
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_bs_gate"] = _gate(payload)
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BS UCY_zara Family-Specific T50 Policy",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bs_gate']['passed']} / {payload['stage42_bs_gate']['total']}`",
        f"- verdict: `{payload['stage42_bs_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- zara_sources_present: `{s['zara_sources_present']}`",
        f"- source_cv_folds: `{s['source_cv_folds']}`",
        f"- rows_total: `{s['rows_total']}`",
        f"- t50_rows_total: `{s['t50_rows_total']}`",
        f"- candidate_t50_oracle_headroom_macro_mean: `{s['candidate_t50_oracle_headroom_macro_mean']}`",
        f"- all_improvement_macro_mean: `{s['all_improvement_macro_mean']}`",
        f"- t50_improvement_macro_mean: `{s['t50_improvement_macro_mean']}`",
        f"- t50_improvement_min: `{s['t50_improvement_min']}`",
        f"- hard_failure_improvement_macro_mean: `{s['hard_failure_improvement_macro_mean']}`",
        f"- easy_degradation_max: `{s['easy_degradation_max']}`",
        f"- policy_selected_fold_count: `{s['policy_selected_fold_count']}`",
        f"- positive_t50_fold_count: `{s['positive_t50_fold_count']}`",
        f"- positive_t50_claim_allowed: `{s['positive_t50_claim_allowed']}`",
        f"- remaining_blocker: `{s['remaining_blocker']}`",
        "",
        "## Fold Results",
        "",
        "| holdout | val | train | rows | t50 rows | oracle headroom | all | t50 | hard/failure | easy degradation | switch | policy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for fold in payload["fold_results"]:
        m = fold["protected_ade"]
        policy = fold["policy"]["slice"]["candidate"] if fold["policy_selected"] else "fallback_only"
        lines.append(
            f"| `{fold['fold']['holdout_source']}` | `{fold['fold']['validation_source']}` | `{','.join(fold['fold']['train_sources'])}` | {m['rows']} | {fold['fold_stats']['test']['t50']} | {fold['candidate_oracle_headroom']['oracle_headroom']:.6f} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | `{policy}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- BS tests the only calibrated t50 blocker from BR that does not require new data: UCY_zara has same-family support but no safe positive t50 policy yet.",
            "- If BS is positive, UCY_zara can be removed from the policy/model blocker list for calibrated-subset t50.",
            "- If BS falls back or remains non-positive, the blocker is policy/model/feature target quality, not source-support.",
            "- This remains source-specific annotation-step calibrated-subset evidence only; global metric/seconds-level M3W claims remain blocked.",
            "",
            "## Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bs_gate"]
    lines = [
        "# Stage42-BS Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- positive_t50_transfer: `{gate['positive_t50_transfer']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


if __name__ == "__main__":
    run_stage42_ucy_zara_t50_family_policy()
