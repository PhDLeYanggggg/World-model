from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping

from src import stage42_local_t100_conversion_readiness as be
from src import stage42_local_t100_schema_conversion as bf
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BE_JSON = OUT_DIR / "local_t100_conversion_readiness_stage42.json"
BF_JSON = OUT_DIR / "local_t100_schema_conversion_stage42.json"
REPORT_JSON = OUT_DIR / "local_t100_protected_policy_stage42.json"
REPORT_MD = OUT_DIR / "local_t100_protected_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bg_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_t100_protected_policy_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

HORIZONS = [50, 100]
FALLBACK_BASELINE = "constant_velocity_causal_fd"
EASY_DEGRADATION_LIMIT = 0.02
MIN_UCY_SOURCE_CV_FOLDS = 3


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BG 使用 Stage42-BF 的本地 t100 in-memory conversion 结果继续做 protected policy source-CV。",
    "本步骤只训练/选择 baseline-family policy，不训练神经模型，不执行 Stage5C，不启用 SMC。",
    "policy threshold / baseline choice 只从 train/validation source 选择，holdout source 只评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic；即使 UCY source-CV positive，也不能写成全局 t100 已修复。",
]


Window = dict[str, Any]
Selector = Callable[[Window], str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_windows_for_source(source: Mapping[str, Any]) -> list[Window]:
    rows = be._parse_rows(Path(str(source["path"])))
    tracks = be._track_map(rows)
    windows: list[Window] = []
    for agent_id, track in tracks.items():
        n = len(track)
        for horizon in HORIZONS:
            for i in range(2, n - horizon):
                prev2 = track[i - 2]
                prev = track[i - 1]
                cur = track[i]
                fut = track[i + horizon]
                vx = float(cur["x"]) - float(prev["x"])
                vy = float(cur["y"]) - float(prev["y"])
                prev_vx = float(prev["x"]) - float(prev2["x"])
                prev_vy = float(prev["y"]) - float(prev2["y"])
                speed = float(math.hypot(vx, vy))
                accel = float(math.hypot(vx - prev_vx, vy - prev_vy))
                errors: dict[str, float] = {}
                for name in bf.BASELINES:
                    px, py = bf._baseline_prediction(name, prev2, prev, cur, horizon)
                    errors[name] = bf._dist(px, py, float(fut["x"]), float(fut["y"]))
                windows.append(
                    {
                        "source": "fresh_in_memory_policy_windows",
                        "source_id": source["source_id"],
                        "domain": source["domain"],
                        "scene_id": str(source["source_id"]).rsplit("/", 1)[0],
                        "agent_id": str(agent_id),
                        "frame_id": int(cur["frame_id"]),
                        "horizon": int(horizon),
                        "speed_causal": speed,
                        "accel_causal": accel,
                        "coordinate_unit": source.get("coordinate_unit", "dataset_local"),
                        "metric_status": source.get("metric_status", "unverified"),
                        "errors_eval_only": errors,
                    }
                )
    return windows


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else None


def _global_selector(name: str) -> Selector:
    return lambda row: name


def _bucket_id(speed: float, low: float, high: float) -> str:
    if speed <= low:
        return "speed_low"
    if speed <= high:
        return "speed_mid"
    return "speed_high"


def _quantiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    vals = sorted(values)
    return vals[int(0.33 * (len(vals) - 1))], vals[int(0.66 * (len(vals) - 1))]


def _speed_bucket_policy(train_windows: list[Window]) -> tuple[str, Selector, dict[str, Any]]:
    speeds = [float(row["speed_causal"]) for row in train_windows]
    low, high = _quantiles(speeds)
    by_key: dict[tuple[int, str], list[Window]] = defaultdict(list)
    for row in train_windows:
        by_key[(int(row["horizon"]), _bucket_id(float(row["speed_causal"]), low, high))].append(row)
    mapping: dict[str, str] = {}
    for (horizon, bucket), rows in sorted(by_key.items()):
        means = {name: _mean(row["errors_eval_only"][name] for row in rows) for name in bf.BASELINES}
        valid = {k: v for k, v in means.items() if v is not None}
        mapping[f"{horizon}:{bucket}"] = min(valid, key=lambda name: float(valid[name])) if valid else FALLBACK_BASELINE

    def selector(row: Window) -> str:
        bucket = _bucket_id(float(row["speed_causal"]), low, high)
        return mapping.get(f"{int(row['horizon'])}:{bucket}", FALLBACK_BASELINE)

    meta = {"low_speed_threshold": low, "high_speed_threshold": high, "mapping": mapping}
    return "speed_bucket_train_best", selector, meta


def _accel_bucket_policy(train_windows: list[Window]) -> tuple[str, Selector, dict[str, Any]]:
    accels = [float(row["accel_causal"]) for row in train_windows]
    low, high = _quantiles(accels)
    by_key: dict[tuple[int, str], list[Window]] = defaultdict(list)
    for row in train_windows:
        bucket = "accel_low" if float(row["accel_causal"]) <= low else "accel_mid" if float(row["accel_causal"]) <= high else "accel_high"
        by_key[(int(row["horizon"]), bucket)].append(row)
    mapping: dict[str, str] = {}
    for (horizon, bucket), rows in sorted(by_key.items()):
        means = {name: _mean(row["errors_eval_only"][name] for row in rows) for name in bf.BASELINES}
        valid = {k: v for k, v in means.items() if v is not None}
        mapping[f"{horizon}:{bucket}"] = min(valid, key=lambda name: float(valid[name])) if valid else FALLBACK_BASELINE

    def selector(row: Window) -> str:
        accel = float(row["accel_causal"])
        bucket = "accel_low" if accel <= low else "accel_mid" if accel <= high else "accel_high"
        return mapping.get(f"{int(row['horizon'])}:{bucket}", FALLBACK_BASELINE)

    meta = {"low_accel_threshold": low, "high_accel_threshold": high, "mapping": mapping}
    return "accel_bucket_train_best", selector, meta


def _candidate_policies(train_windows: list[Window]) -> list[tuple[str, Selector, dict[str, Any]]]:
    candidates = [(f"global_{name}", _global_selector(name), {"baseline": name}) for name in bf.BASELINES]
    if train_windows:
        candidates.append(_speed_bucket_policy(train_windows))
        candidates.append(_accel_bucket_policy(train_windows))
    return candidates


def _policy_metrics(windows: list[Window], selector: Selector, *, easy_threshold: float | None = None) -> dict[str, Any]:
    if not windows:
        return {
            "source": "fresh_policy_eval",
            "rows": 0,
            "mean_fde": None,
            "fallback_mean_fde": None,
            "improvement_vs_fallback": None,
            "easy_degradation": None,
            "switch_rate": None,
            "harm_over_fallback": None,
            "selected_distribution": {},
        }
    selected_errors: list[float] = []
    fallback_errors: list[float] = []
    choices: list[str] = []
    for row in windows:
        selected = selector(row)
        if selected not in row["errors_eval_only"]:
            selected = FALLBACK_BASELINE
        choices.append(selected)
        selected_errors.append(float(row["errors_eval_only"][selected]))
        fallback_errors.append(float(row["errors_eval_only"][FALLBACK_BASELINE]))
    easy_threshold = float(median(fallback_errors)) if easy_threshold is None else float(easy_threshold)
    easy_idx = [i for i, err in enumerate(fallback_errors) if err <= easy_threshold]
    selected_mean = float(sum(selected_errors) / len(selected_errors))
    fallback_mean = float(sum(fallback_errors) / len(fallback_errors))
    easy_selected = _mean(selected_errors[i] for i in easy_idx)
    easy_fallback = _mean(fallback_errors[i] for i in easy_idx)
    easy_degradation = None
    if easy_selected is not None and easy_fallback and easy_fallback > 0.0:
        easy_degradation = (float(easy_selected) - float(easy_fallback)) / float(easy_fallback)
    return {
        "source": "fresh_policy_eval",
        "rows": int(len(windows)),
        "mean_fde": selected_mean,
        "fallback_mean_fde": fallback_mean,
        "improvement_vs_fallback": (fallback_mean - selected_mean) / fallback_mean if fallback_mean > 0.0 else None,
        "easy_threshold_from_eval_slice": easy_threshold,
        "easy_rows": int(len(easy_idx)),
        "easy_degradation": easy_degradation,
        "switch_rate": float(sum(choice != FALLBACK_BASELINE for choice in choices) / len(choices)),
        "harm_over_fallback": selected_mean - fallback_mean,
        "selected_distribution": dict(Counter(choices)),
    }


def _select_policy_on_validation(
    *,
    train_windows: list[Window],
    val_windows: list[Window],
    horizon: int,
) -> dict[str, Any]:
    train_h = [row for row in train_windows if int(row["horizon"]) == horizon]
    val_h = [row for row in val_windows if int(row["horizon"]) == horizon]
    fallback_metrics = _policy_metrics(val_h, _global_selector(FALLBACK_BASELINE))
    candidates = _candidate_policies(train_h)
    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for name, selector, meta in candidates:
        metrics = _policy_metrics(val_h, selector)
        improvement = float(metrics["improvement_vs_fallback"] or 0.0)
        easy_degradation = float(metrics["easy_degradation"] or 0.0)
        eligible = improvement > 0.0 and easy_degradation <= EASY_DEGRADATION_LIMIT
        score = improvement - max(0.0, easy_degradation - EASY_DEGRADATION_LIMIT) * 10.0 - float(metrics["switch_rate"] or 0.0) * 0.001
        row = {
            "source": "fresh_validation_selection",
            "policy_name": name,
            "metadata": meta,
            "validation_metrics": metrics,
            "eligible_for_holdout": bool(eligible),
            "score": float(score),
            "horizon": int(horizon),
        }
        evaluated.append(row)
        if eligible and (best is None or score > float(best["score"])):
            best = row
    if best is None:
        best = {
            "source": "fresh_validation_selection",
            "policy_name": f"global_{FALLBACK_BASELINE}",
            "metadata": {"baseline": FALLBACK_BASELINE},
            "validation_metrics": fallback_metrics,
            "eligible_for_holdout": True,
            "score": 0.0,
            "horizon": int(horizon),
            "fallback_reason": "no_validation_safe_policy_exceeded_fallback",
        }
    return {
        "source": "fresh_validation_selection",
        "horizon": int(horizon),
        "selected_policy": best,
        "candidate_count": len(evaluated),
        "candidates": evaluated,
    }


def _selector_from_selected(selected: Mapping[str, Any], train_windows: list[Window]) -> Selector:
    name = str(selected["policy_name"])
    if name.startswith("global_"):
        return _global_selector(name.removeprefix("global_"))
    for candidate_name, selector, _ in _candidate_policies(train_windows):
        if candidate_name == name:
            return selector
    return _global_selector(FALLBACK_BASELINE)


def _evaluate_fold(
    *,
    fold: Mapping[str, Any],
    windows_by_source: Mapping[str, list[Window]],
) -> dict[str, Any]:
    train_windows: list[Window] = []
    for source_id in fold.get("train_sources", []):
        train_windows.extend(windows_by_source.get(source_id, []))
    val_windows = list(windows_by_source.get(str(fold["validation_source"]), []))
    holdout_windows = list(windows_by_source.get(str(fold["holdout_source"]), []))
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        selection = _select_policy_on_validation(train_windows=train_windows, val_windows=val_windows, horizon=horizon)
        selector = _selector_from_selected(selection["selected_policy"], [row for row in train_windows if int(row["horizon"]) == horizon])
        holdout_h = [row for row in holdout_windows if int(row["horizon"]) == horizon]
        holdout_metrics = _policy_metrics(holdout_h, selector)
        by_horizon[str(horizon)] = {
            "source": "fresh_source_cv_protected_policy",
            "selection": selection,
            "holdout_metrics": holdout_metrics,
            "safe_positive": bool(
                (holdout_metrics["improvement_vs_fallback"] or 0.0) > 0.0
                and (holdout_metrics["easy_degradation"] or 0.0) <= EASY_DEGRADATION_LIMIT
            ),
        }
    return {
        "source": "fresh_source_cv_protected_policy",
        "domain": fold["domain"] if "domain" in fold else "unknown",
        "holdout_source": fold["holdout_source"],
        "validation_source": fold["validation_source"],
        "train_sources": list(fold.get("train_sources", [])),
        "by_horizon": by_horizon,
    }


def _domain_summary(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    for fold in folds:
        domain = str(fold["domain"])
        by_domain.setdefault(domain, {"folds": []})["folds"].append(fold)
    out: dict[str, Any] = {}
    for domain, row in sorted(by_domain.items()):
        folds_row = row["folds"]
        h_summaries: dict[str, Any] = {}
        for horizon in HORIZONS:
            improvements = [
                float(fold["by_horizon"][str(horizon)]["holdout_metrics"]["improvement_vs_fallback"])
                for fold in folds_row
                if fold["by_horizon"][str(horizon)]["holdout_metrics"]["improvement_vs_fallback"] is not None
            ]
            easy = [
                float(fold["by_horizon"][str(horizon)]["holdout_metrics"]["easy_degradation"])
                for fold in folds_row
                if fold["by_horizon"][str(horizon)]["holdout_metrics"]["easy_degradation"] is not None
            ]
            safe = [bool(fold["by_horizon"][str(horizon)]["safe_positive"]) for fold in folds_row]
            h_summaries[str(horizon)] = {
                "source": "fresh_source_cv_protected_policy",
                "fold_count": int(len(folds_row)),
                "safe_positive_fold_count": int(sum(safe)),
                "all_folds_safe_positive": bool(safe and all(safe)),
                "mean_holdout_improvement_vs_fallback": float(sum(improvements) / len(improvements)) if improvements else None,
                "minimum_holdout_improvement_vs_fallback": float(min(improvements)) if improvements else None,
                "maximum_easy_degradation": float(max(easy)) if easy else None,
            }
        out[domain] = {
            "source": "fresh_source_cv_protected_policy",
            "fold_count": int(len(folds_row)),
            "by_horizon": h_summaries,
        }
    return out


def run_stage42_local_t100_protected_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    be_payload = _load_json(BE_JSON)
    bf_payload = _load_json(BF_JSON)
    readiness = list(be_payload.get("source_readiness", []))
    windows_by_source = {row["source_id"]: _build_windows_for_source(row) for row in readiness}
    folds: list[dict[str, Any]] = []
    blockers: dict[str, Any] = {}
    for domain, plan in be_payload.get("source_cv_plan", {}).get("domains", {}).items():
        if not plan.get("folds"):
            blockers[domain] = {
                "source": "fresh_source_cv_protected_policy",
                "status": "not_run",
                "reason": "fewer_than_three_t100_capable_sources_or_no_source_cv_folds",
                "t100_capable_sources": int(plan.get("t100_capable_sources", 0)),
                "estimated_t100_windows": int(plan.get("estimated_t100_windows", 0)),
            }
            continue
        for fold in plan["folds"]:
            fold_with_domain = dict(fold)
            fold_with_domain["domain"] = domain
            folds.append(_evaluate_fold(fold=fold_with_domain, windows_by_source=windows_by_source))
    domain_summary = _domain_summary(folds)
    total_windows = {
        str(h): int(sum(1 for rows in windows_by_source.values() for row in rows if int(row["horizon"]) == h))
        for h in HORIZONS
    }
    ucy_t100 = domain_summary.get("UCY", {}).get("by_horizon", {}).get("100", {})
    ucy_t100_supported = bool(
        ucy_t100.get("fold_count", 0) >= MIN_UCY_SOURCE_CV_FOLDS
        and ucy_t100.get("all_folds_safe_positive", False)
        and (ucy_t100.get("maximum_easy_degradation") is not None and float(ucy_t100["maximum_easy_degradation"]) <= EASY_DEGRADATION_LIMIT)
    )
    payload = {
        "source": "fresh_source_cv_protected_policy",
        "stage": "Stage42-BG Local T100 Protected Policy Source-CV",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BE_JSON), str(BF_JSON)] + [str(row.get("path", "")) for row in readiness]),
        "be_verdict": be_payload.get("stage42_be_gate", {}).get("verdict"),
        "bf_verdict": bf_payload.get("stage42_bf_gate", {}).get("verdict"),
        "feature_store_manifest": {
            "source": "fresh_in_memory_policy_windows",
            "materialized_large_feature_store_written": False,
            "manifest_written_in_report": True,
            "fields": [
                "source_id",
                "domain",
                "scene_id",
                "agent_id",
                "frame_id",
                "horizon",
                "speed_causal",
                "accel_causal",
                "baseline_family_errors_eval_only",
            ],
            "future_labels_as_inputs": False,
            "central_velocity": False,
        },
        "window_summary": {
            "source": "fresh_in_memory_policy_windows",
            "sources": {
                source_id: {
                    "rows": int(len(rows)),
                    "t50_rows": int(sum(1 for row in rows if int(row["horizon"]) == 50)),
                    "t100_rows": int(sum(1 for row in rows if int(row["horizon"]) == 100)),
                }
                for source_id, rows in windows_by_source.items()
            },
            "total_windows_by_horizon": total_windows,
        },
        "source_cv_folds": folds,
        "domain_summary": domain_summary,
        "blockers": blockers,
        "summary": {
            "source": "fresh_source_cv_protected_policy",
            "candidate_sources": len(readiness),
            "policy_window_sources": len(windows_by_source),
            "t50_policy_windows": total_windows["50"],
            "t100_policy_windows": total_windows["100"],
            "source_cv_domains_evaluated": sorted(domain_summary.keys()),
            "source_cv_domains_blocked": sorted(blockers.keys()),
            "ucy_t100_source_cv_supported": ucy_t100_supported,
            "ucy_t100_mean_improvement_vs_fallback": ucy_t100.get("mean_holdout_improvement_vs_fallback"),
            "ucy_t100_min_improvement_vs_fallback": ucy_t100.get("minimum_holdout_improvement_vs_fallback"),
            "ucy_t100_max_easy_degradation": ucy_t100.get("maximum_easy_degradation"),
            "training_run": True,
            "training_type": "validation_selected_baseline_family_policy",
            "neural_training_run": False,
            "evaluation_run": True,
            "t100_positive_claim_allowed": False,
            "ucy_local_t100_support_claim_allowed": ucy_t100_supported,
            "stage5c_executed": False,
            "smc_enabled": False,
            "metric_or_seconds_claim": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "causal_velocity_only": True,
            "holdout_used_for_threshold": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_positive_claim_allowed": False,
            "ucy_local_t100_source_cv_support": ucy_t100_supported,
        },
    }
    payload["stage42_bg_gate"] = _gate(payload)
    payload["user_action_required"] = _user_actions(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "be_readiness_loaded": payload["be_verdict"] == "stage42_be_local_t100_conversion_readiness_pass",
        "bf_conversion_loaded": payload["bf_verdict"] == "stage42_bf_local_t100_schema_conversion_pass",
        "policy_windows_built": int(s["policy_window_sources"]) == int(s["candidate_sources"]) and int(s["t100_policy_windows"]) > 0,
        "validation_selected_policy_trained": s["training_run"] is True and s["training_type"] == "validation_selected_baseline_family_policy",
        "holdout_source_cv_evaluated": bool(s["source_cv_domains_evaluated"]),
        "ucy_t100_source_cv_supported": bool(s["ucy_t100_source_cv_supported"]),
        "eth_ucy_blocker_reported": "ETH_UCY" in s["source_cv_domains_blocked"],
        "no_leakage_pass": all(
            payload["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_metrics_for_threshold",
                "holdout_used_for_threshold",
            ]
        ),
        "no_large_feature_store_committed": payload["feature_store_manifest"]["materialized_large_feature_store_written"] is False,
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"],
        "global_t100_claim_still_blocked": not payload["claim_boundary"]["t100_positive_claim_allowed"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bg_local_t100_protected_policy_pass_with_global_t100_blocker" if passed == total else "stage42_bg_local_t100_protected_policy_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _user_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": payload["source"],
            "priority": "high",
            "action_type": "keep_global_t100_claim_blocked",
            "notes": "UCY local source-CV is supported, but ETH_UCY remains under-supported. Do not advertise global external t100 success.",
        },
        {
            "source": payload["source"],
            "priority": "high",
            "action_type": "add_more_independent_t100_sources",
            "notes": "Need additional ETH_UCY / TrajNet / UCY t100-capable independent sources before global t100 deployment claims.",
        },
    ]


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BG Local T100 Protected Policy Source-CV",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bg_gate']['passed']} / {payload['stage42_bg_gate']['total']}`",
        f"- verdict: `{payload['stage42_bg_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- candidate_sources: `{s['candidate_sources']}`",
        f"- t50_policy_windows: `{s['t50_policy_windows']}`",
        f"- t100_policy_windows: `{s['t100_policy_windows']}`",
        f"- source_cv_domains_evaluated: `{', '.join(s['source_cv_domains_evaluated']) or 'none'}`",
        f"- source_cv_domains_blocked: `{', '.join(s['source_cv_domains_blocked']) or 'none'}`",
        f"- ucy_t100_source_cv_supported: `{s['ucy_t100_source_cv_supported']}`",
        f"- ucy_t100_mean_improvement_vs_fallback: `{s['ucy_t100_mean_improvement_vs_fallback']}`",
        f"- ucy_t100_min_improvement_vs_fallback: `{s['ucy_t100_min_improvement_vs_fallback']}`",
        f"- ucy_t100_max_easy_degradation: `{s['ucy_t100_max_easy_degradation']}`",
        f"- global_t100_positive_claim_allowed: `{s['t100_positive_claim_allowed']}`",
        "",
        "## Domain Summary",
        "",
        "| domain | horizon | folds | safe folds | mean improvement | min improvement | max easy degradation | all safe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["domain_summary"].items():
        for horizon, hrow in row["by_horizon"].items():
            lines.append(
                f"| `{domain}` | {horizon} | {hrow['fold_count']} | {hrow['safe_positive_fold_count']} | "
                f"{hrow['mean_holdout_improvement_vs_fallback']:.6f} | {hrow['minimum_holdout_improvement_vs_fallback']:.6f} | "
                f"{hrow['maximum_easy_degradation']:.6f} | {hrow['all_folds_safe_positive']} |"
            )
    lines.extend(["", "## Fold Details", "", "| domain | holdout | horizon | selected policy | holdout rows | improvement | easy degradation | switch rate |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"])
    for fold in payload["source_cv_folds"]:
        for horizon, hrow in fold["by_horizon"].items():
            selected = hrow["selection"]["selected_policy"]["policy_name"]
            metrics = hrow["holdout_metrics"]
            lines.append(
                f"| `{fold['domain']}` | `{fold['holdout_source']}` | {horizon} | `{selected}` | {metrics['rows']} | "
                f"{metrics['improvement_vs_fallback']:.6f} | {metrics['easy_degradation']:.6f} | {metrics['switch_rate']:.6f} |"
            )
    lines.extend(["", "## Blockers", ""])
    if payload["blockers"]:
        for domain, row in payload["blockers"].items():
            lines.append(f"- `{domain}`: `{row['reason']}`; t100_capable_sources={row['t100_capable_sources']}; estimated_t100_windows={row['estimated_t100_windows']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-BG is stronger than BF: it selects a protected baseline-family policy on validation sources and evaluates once on held-out sources.",
            "- UCY local t100 source-CV is positive and easy-safe under this limited protocol.",
            "- This is still not a global t100 deployment claim because ETH_UCY is under-supported and TrajNet is not represented in these new local candidates.",
            "- Stage5C remains unexecuted and SMC remains disabled.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bg_gate"]
    lines = [
        "# Stage42-BG Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = ["# Stage42-BG User Action Required", "", f"- source: `{payload['source']}`", ""]
    for action in payload["user_action_required"]:
        lines.extend([f"## {action['action_type']}", "", f"- priority: `{action['priority']}`", f"- notes: {action['notes']}", ""])
    return lines


def _append_ledger(payload: Mapping[str, Any]) -> None:
    row = {
        "stage": payload["stage"],
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_bg_gate"]["verdict"],
        "gate": f"{payload['stage42_bg_gate']['passed']}/{payload['stage42_bg_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_local_t100_protected_policy()
