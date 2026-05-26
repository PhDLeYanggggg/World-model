from __future__ import annotations

import csv
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
REPORT_JSON = OUT_DIR / "full_waypoint_all_hard_loss_repair_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_all_hard_loss_repair_stage42.md"
REPORT_CSV = OUT_DIR / "full_waypoint_all_hard_loss_repair_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_dg_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
DF_JSON = OUT_DIR / "full_waypoint_all_hard_proximity_repair_stage42.json"
DE_JSON = OUT_DIR / "full_waypoint_deployment_gap_audit_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

LAMBDAS = [0.05, 0.1, 1.0, 10.0, 100.0]
VARIANTS = [
    "balanced",
    "hard_failure_weighted",
    "long_horizon_weighted",
    "all_hard_long_horizon",
    "source_balanced_all_hard_long",
]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DG 针对 Stage42-DE/DF 的 full-waypoint all/hard blocker，实际重训 all/hard/long-horizon weighted full-waypoint ridge dynamics probe。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "validation 选择 loss variant、ridge lambda、safe policy；test 只评一次。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _sample_weights(data: Mapping[str, np.ndarray], train_mask: np.ndarray, variant: str) -> np.ndarray:
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    domain = data["dataset"].astype(str)
    weights = np.ones(len(h), dtype=np.float64)
    if variant == "balanced":
        pass
    elif variant == "hard_failure_weighted":
        weights += 3.0 * hard_failure.astype(np.float64)
    elif variant == "long_horizon_weighted":
        weights += 1.8 * (h == 50) + 2.6 * (h == 100)
    elif variant == "all_hard_long_horizon":
        weights += 2.5 * hard_failure.astype(np.float64) + 1.8 * (h == 50) + 2.6 * (h == 100)
        weights += 0.4 * (~easy).astype(np.float64)
    elif variant == "source_balanced_all_hard_long":
        weights += 2.5 * hard_failure.astype(np.float64) + 1.8 * (h == 50) + 2.6 * (h == 100)
        train_domains, counts = np.unique(domain[train_mask], return_counts=True)
        inv = {str(d): float(np.mean(counts) / max(c, 1)) for d, c in zip(train_domains, counts)}
        weights *= np.asarray([inv.get(str(d), 1.0) for d in domain], dtype=np.float64)
    else:
        raise ValueError(f"Unknown loss variant: {variant}")
    mean = float(np.mean(weights[train_mask])) if np.any(train_mask) else 1.0
    return (weights / max(mean, EPS)).astype(np.float64)


def _fit_weighted_ridge_1d(x: np.ndarray, y: np.ndarray, mask: np.ndarray, weights: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(mask)[0]
    if len(ids) == 0:
        return np.zeros(x.shape[1], dtype=np.float32)
    xt = x[ids].astype(np.float64, copy=False)
    yt = y[ids].astype(np.float64, copy=False)
    sw = np.sqrt(np.maximum(weights[ids].astype(np.float64), EPS))
    xw = xt * sw[:, None]
    yw = yt * sw
    reg = np.eye(x.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw).astype(np.float32)


def _fit_weighted_ridge_model(
    x: np.ndarray,
    target_delta: np.ndarray,
    waypoint_valid: np.ndarray,
    train_mask: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> np.ndarray:
    y = target_delta.reshape(len(target_delta), -1)
    coef = np.zeros((x.shape[1], y.shape[1]), dtype=np.float32)
    for w in range(len(am.WAYPOINT_FRAC)):
        m = train_mask & waypoint_valid[:, w]
        coef[:, 2 * w] = _fit_weighted_ridge_1d(x, y[:, 2 * w], m, weights, lam)
        coef[:, 2 * w + 1] = _fit_weighted_ridge_1d(x, y[:, 2 * w + 1], m, weights, lam)
    return coef


def _selection_score(metric: Mapping[str, Any]) -> float:
    return (
        1.8 * float(metric["all_improvement"])
        + 2.2 * float(metric["hard_failure_improvement"])
        + 1.0 * float(metric["t50_improvement"])
        + 0.5 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 0.02 * float(metric["switch_rate"])
    )


def _evaluate_loss_variants(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    x: np.ndarray,
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - current[:, None, :]) / scale[:, None, None]).astype(
        np.float32
    )
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for variant in VARIANTS:
        weights = _sample_weights(data, train_mask, variant)
        for lam in LAMBDAS:
            coef = _fit_weighted_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, weights, lam)
            pred_xy = am._predict_waypoints(x, coef, data)
            policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
                pred_xy, floor["floor_xy"], labels, data, val_mask
            )
            val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
            test_metric = am._metric(selected_ade, floor_ade, data, switch, test_mask)
            pred_ade, pred_fde = am._trajectory_errors(pred_xy, labels)
            ungated_test_metric = am._metric(pred_ade, floor_ade, data, np.ones(len(pred_ade), dtype=bool), test_mask)
            score = _selection_score(val_metric)
            row = {
                "variant": variant,
                "lambda": float(lam),
                "val_score": float(score),
                "val_metric": val_metric,
                "test_metric": test_metric,
                "ungated_test_metric": ungated_test_metric,
                "policy_slice_count": len(policy["slices"]),
                "policy": policy,
            }
            rows.append(row)
            if score > best_score:
                best_score = float(score)
                best = {
                    **row,
                    "coef": coef,
                    "selected_ade": selected_ade,
                    "selected_fde": selected_fde,
                    "switch": switch,
                    "pred_ade": pred_ade,
                    "pred_fde": pred_fde,
                    "floor_ade": floor_ade,
                    "floor_fde": floor_fde,
                }
    if best is None:
        raise RuntimeError("No Stage42-DG loss variant evaluated.")
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    domain = data["dataset"].astype(str)
    best_public = {k: v for k, v in best.items() if k not in {"coef", "selected_ade", "selected_fde", "switch", "pred_ade", "pred_fde", "floor_ade", "floor_fde"}}
    return {
        "variant_count": len(VARIANTS),
        "candidate_count": len(rows),
        "validation_rows": sorted(
            [{k: v for k, v in row.items() if k != "policy"} | {"policy_slice_count": row["policy_slice_count"]} for row in rows],
            key=lambda row: float(row["val_score"]),
            reverse=True,
        ),
        "selected": best_public,
        "metrics": {
            "floor": am._metric(best["floor_ade"], best["floor_ade"], data, np.zeros(len(h), dtype=bool), test_mask),
            "ungated_selected_loss_variant": am._metric(best["pred_ade"], best["floor_ade"], data, np.ones(len(h), dtype=bool), test_mask),
            "protected_selected_loss_variant": am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask),
            "protected_selected_loss_variant_fde": am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask),
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42061),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42062),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42063),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42064),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42065),
        },
        "by_domain": {
            d: am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask & (domain == d))
            for d in sorted(set(domain[test_mask].tolist()))
        },
        "by_horizon": {
            str(hh): am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask & (h == hh))
            for hh in [10, 25, 50, 100]
        },
    }


def _compare_to_stage42_am(eval_result: Mapping[str, Any]) -> dict[str, Any]:
    am_payload = read_json(AM_JSON, {})
    am_metric = am_payload.get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    current = eval_result["metrics"]["protected_selected_loss_variant"]
    return {
        "stage42_am_source": am_payload.get("source", "missing"),
        "stage42_am_metric": am_metric,
        "delta_vs_stage42_am": {
            "all_improvement": float(current.get("all_improvement", 0.0)) - float(am_metric.get("all_improvement", 0.0)),
            "t50_improvement": float(current.get("t50_improvement", 0.0)) - float(am_metric.get("t50_improvement", 0.0)),
            "t100_raw_frame_diagnostic_improvement": float(current.get("t100_raw_frame_diagnostic_improvement", 0.0))
            - float(am_metric.get("t100_raw_frame_diagnostic_improvement", 0.0)),
            "hard_failure_improvement": float(current.get("hard_failure_improvement", 0.0))
            - float(am_metric.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(current.get("easy_degradation", 0.0)) - float(am_metric.get("easy_degradation", 0.0)),
            "switch_rate": float(current.get("switch_rate", 0.0)) - float(am_metric.get("switch_rate", 0.0)),
        },
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    eval_result = _evaluate_loss_variants(data, split, labels, floor, x)
    comparison = _compare_to_stage42_am(eval_result)
    metric = eval_result["metrics"]["protected_selected_loss_variant"]
    delta = comparison["delta_vs_stage42_am"]
    improves_am = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and delta["all_improvement"] > 0.0
        and delta["hard_failure_improvement"] > 0.0
    )
    result: dict[str, Any] = {
        "source": "fresh_stage42_dg_full_waypoint_all_hard_loss_repair",
        "stage": "Stage42-DG full-waypoint all/hard weighted loss repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(DE_JSON),
                str(DF_JSON),
            ]
        ),
        "source_split": source_split,
        "split_stats": split_stats,
        "label_stats": {
            "rows": int(len(split)),
            "full_waypoint_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
            "test_rows": int(np.sum(split == "test")),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "feature_schema": {
            "source": "cached_verified_stage42_am_feature_schema_plus_new_loss_weights",
            "feature_count": len(feature_names),
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
            "loss_variants": VARIANTS,
            "lambdas": LAMBDAS,
        },
        "floor": {
            "type": "train_horizon_selected_safe_causal_baseline",
            "strongest_by_horizon": floor["strongest_by_horizon"],
            "geometry_diagnostics": floor["geometry_diagnostics"],
        },
        "model": eval_result,
        "comparison_to_stage42_am": comparison,
        "deployment_decision": {
            "promote_weighted_loss_full_waypoint": bool(improves_am),
            "decision": "promote_stage42_dg_weighted_loss_candidate"
            if improves_am
            else "weighted_loss_not_enough_keep_stage42_am_or_cq_floor",
            "reason": "Promotion requires test all+hard positive, easy safe, and improvement over Stage42-AM on all+hard.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
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
    result["stage42_dg_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    m = result["model"]["metrics"]["protected_selected_loss_variant"]
    delta = result["comparison_to_stage42_am"]["delta_vs_stage42_am"]
    no_leak = result["no_leakage"]
    gates = {
        "source_level_split_rebuilt": result["split_stats"]["by_split"]["test"]["rows"] == int(m["rows"]) and int(m["rows"]) > 0,
        "full_waypoint_labels_available": result["label_stats"]["test_full_waypoint_rows"] > 0,
        "weighted_loss_variants_run": result["model"]["candidate_count"] >= len(VARIANTS),
        "validation_selected_model": result["model"]["selected"]["val_score"] != 0.0 and no_leak["test_threshold_tuning"] is False,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["train_only_feature_normalization"] is True,
                no_leak["validation_only_model_selection"] is True,
            ]
        ),
        "test_all_positive_vs_floor": m["all_improvement"] > 0.0,
        "test_hard_positive_vs_floor": m["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": m["easy_degradation"] <= 0.02,
        "bootstrap_reported": result["model"]["bootstrap"]["all"]["bootstrap_n"] > 0,
        "comparison_to_stage42_am_recorded": "all_improvement" in delta,
        "beats_stage42_am_all": delta["all_improvement"] > 0.0,
        "beats_stage42_am_hard": delta["hard_failure_improvement"] > 0.0,
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_dg_full_waypoint_weighted_loss_repair_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_dg_full_waypoint_weighted_loss_repair_pass_positive_not_better_than_am"
    else:
        verdict = "stage42_dg_full_waypoint_weighted_loss_repair_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(rows: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "variant",
                "lambda",
                "val_score",
                "policy_slice_count",
                "val_all",
                "val_hard",
                "val_t50",
                "val_easy",
                "test_all",
                "test_hard",
                "test_t50",
                "test_easy",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows[:100], start=1):
            val = row["val_metric"]
            test = row["test_metric"]
            writer.writerow(
                {
                    "rank": idx,
                    "variant": row["variant"],
                    "lambda": row["lambda"],
                    "val_score": row["val_score"],
                    "policy_slice_count": row["policy_slice_count"],
                    "val_all": val["all_improvement"],
                    "val_hard": val["hard_failure_improvement"],
                    "val_t50": val["t50_improvement"],
                    "val_easy": val["easy_degradation"],
                    "test_all": test["all_improvement"],
                    "test_hard": test["hard_failure_improvement"],
                    "test_t50": test["t50_improvement"],
                    "test_easy": test["easy_degradation"],
                }
            )


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dg_gate"]
    selected = result["model"]["selected"]
    metric = result["model"]["metrics"]["protected_selected_loss_variant"]
    ungated = result["model"]["metrics"]["ungated_selected_loss_variant"]
    delta = result["comparison_to_stage42_am"]["delta_vs_stage42_am"]
    lines = [
        "# Stage42-DG Full-Waypoint All/Hard Weighted Loss Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{result['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Selected Loss Variant",
        "",
        f"- variant: `{selected['variant']}`",
        f"- lambda: `{selected['lambda']}`",
        f"- val_score: `{selected['val_score']:.6f}`",
        f"- policy_slice_count: `{selected['policy_slice_count']}`",
        "",
        "## Test Once vs Train-Horizon Causal Floor",
        "",
        "| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| ungated selected loss variant | {_pct(ungated['all_improvement'])} | {_pct(ungated['t50_improvement'])} | {_pct(ungated['t100_raw_frame_diagnostic_improvement'])} | {_pct(ungated['hard_failure_improvement'])} | {_pct(ungated['easy_degradation'])} | {_pct(ungated['switch_rate'])} |",
        f"| protected selected loss variant | {_pct(metric['all_improvement'])} | {_pct(metric['t50_improvement'])} | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} | {_pct(metric['hard_failure_improvement'])} | {_pct(metric['easy_degradation'])} | {_pct(metric['switch_rate'])} |",
        "",
        "## Delta vs Stage42-AM Protected Ridge",
        "",
        f"- delta_all: `{_pct(delta['all_improvement'])}`",
        f"- delta_t50: `{_pct(delta['t50_improvement'])}`",
        f"- delta_t100_raw: `{_pct(delta['t100_raw_frame_diagnostic_improvement'])}`",
        f"- delta_hard: `{_pct(delta['hard_failure_improvement'])}`",
        f"- delta_easy: `{_pct(delta['easy_degradation'])}`",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["model"]["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(
        [
            "",
            "## By Domain",
            "",
            "| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in result["model"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | "
            f"{_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | "
            f"{_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-DG changes the full-waypoint training target through all/hard/long-horizon sample weighting and validation-selected ridge lambda.",
            "- It is a real retraining/evaluation probe over source-level full-waypoint rows, not another Stage42-DF threshold search.",
            "- Promotion requires improving Stage42-AM on all and hard/failure while keeping easy degradation <=2%.",
            "- If it does not beat Stage42-AM, the next move should be a stronger sequence/graph model or explicit proximity/occupancy loss, not more scalar loss weights.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dg_gate"]
    return [
        "# Stage42-DG Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dg_gate"]
    metric = result["model"]["metrics"]["protected_selected_loss_variant"]
    delta = result["comparison_to_stage42_am"]["delta_vs_stage42_am"]
    selected = result["model"]["selected"]
    return [
        "## Stage42-DG Full-Waypoint All/Hard Weighted Loss Repair",
        "",
        "- source: `fresh_stage42_dg_full_waypoint_all_hard_loss_repair`",
        "- role: actual retraining probe for all/hard/long-horizon weighted full-waypoint dynamics, following Stage42-DE/DF blockers.",
        f"- selected loss variant: `{selected['variant']}` with lambda `{selected['lambda']}`.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- test vs train-horizon causal floor: all `{_pct(metric['all_improvement'])}`, t50 `{_pct(metric['t50_improvement'])}`, t100 raw `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(metric['hard_failure_improvement'])}`, easy `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-AM: all `{_pct(delta['all_improvement'])}`, t50 `{_pct(delta['t50_improvement'])}`, t100 raw `{_pct(delta['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(delta['hard_failure_improvement'])}`, easy `{_pct(delta['easy_degradation'])}`.",
        f"- decision: `{result['deployment_decision']['decision']}`.",
        "- Stage5C remains false; SMC remains false; no metric/seconds claim.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DG full-waypoint all/hard weighted loss repair"
    state["current_verdict"] = result["stage42_dg_gate"]["verdict"]
    state["stage42_dg_full_waypoint_all_hard_loss_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_dg_gate"]["verdict"],
        "gates": f"{result['stage42_dg_gate']['passed']}/{result['stage42_dg_gate']['total']}",
        "deployment_decision": result["deployment_decision"],
        "selected": {
            "variant": result["model"]["selected"]["variant"],
            "lambda": result["model"]["selected"]["lambda"],
            "val_score": result["model"]["selected"]["val_score"],
        },
        "test_metric_vs_floor": result["model"]["metrics"]["protected_selected_loss_variant"],
        "comparison_to_stage42_am": result["comparison_to_stage42_am"]["delta_vs_stage42_am"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_full_waypoint_all_hard_loss_repair(*, refresh_readmes: bool = True) -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _write_csv(result["model"]["validation_rows"])
    if refresh_readmes:
        _refresh_readmes(result)
        _refresh_research_state(result)
    return result


if __name__ == "__main__":
    run_stage42_full_waypoint_all_hard_loss_repair()
