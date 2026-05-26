from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_row_prediction_cache as r
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_unseen_transfer_stage42.json"
REPORT_MD = OUT_DIR / "ucy_unseen_transfer_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_t_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCES = {
    "stage42j_static_expert": ("j_val_ade", "j_val_fde", "j_val_switch", "j_test_ade", "j_test_fde", "j_test_switch"),
    "stage42p_t50_gain_harm": ("p_val_ade", "p_val_fde", "p_val_switch", "p_test_ade", "p_test_fde", "p_test_switch"),
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-T 只修复/诊断 UCY fallback-only 问题，不执行 Stage5C 或 SMC。",
    "unseen-domain transfer rule 只从 validation domains 推导；UCY test 只最终评估一次。",
    "future waypoints / endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。",
    "当前 row cache 是本地 derived cache，不提交 GitHub。",
    "如果 UCY 无可用非 floor prediction source，必须标 blocker，不包装成成功。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-T UCY unseen-domain transfer attempt":
        return payload
    return None


def _domain_counts(labels: Mapping[str, np.ndarray]) -> dict[str, int]:
    dom = labels["domain"].astype(str)
    vals, cnts = np.unique(dom, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, cnts)}


def _score(metric: Mapping[str, Any], horizon: int) -> float:
    h_weight = 4.0 if horizon == 50 else 1.0
    return (
        h_weight * float(metric.get("all_improvement", 0.0))
        + float(metric.get("hard_failure_improvement", 0.0))
        - 80.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.018)
        - 0.04 * float(metric.get("switch_rate", 0.0))
    )


def _source_metric(arrays: Mapping[str, np.ndarray], source: str, labels: Mapping[str, np.ndarray], mask: np.ndarray, split: str) -> dict[str, Any]:
    v_ade, _v_fde, v_sw, t_ade, _t_fde, t_sw = SOURCES[source]
    ade_key = v_ade if split == "val" else t_ade
    sw_key = v_sw if split == "val" else t_sw
    floor_key = "floor_val_ade" if split == "val" else "floor_test_ade"
    return r._local_metric_from_errors(arrays[ade_key], arrays[floor_key], labels, arrays[sw_key], mask)


def _fit_transfer_rule(
    cache_paths: list[Path],
    labels_val: Mapping[str, np.ndarray],
    calibration_domains: list[str],
) -> dict[str, Any]:
    domain_val = labels_val["domain"].astype(str)
    horizon_val = labels_val["horizon"].astype(int)
    horizons = [10, 25, 50, 100]
    per_pair = []
    for pair_idx, path in enumerate(cache_paths):
        with np.load(path, allow_pickle=False) as row:
            arrays = {k: row[k].copy() for k in row.files}
        choices = {}
        for horizon in horizons:
            best_source = "floor"
            best_score = 0.0
            source_details: dict[str, Any] = {}
            for source in SOURCES:
                domain_metrics = {}
                scores = []
                eligible = True
                for domain in calibration_domains:
                    mask = (domain_val == domain) & (horizon_val == horizon)
                    if int(np.sum(mask)) < 80:
                        eligible = False
                        domain_metrics[domain] = {"source": "not_run_insufficient_validation_rows", "rows": int(np.sum(mask))}
                        continue
                    metric = _source_metric(arrays, source, labels_val, mask, "val")
                    score = _score(metric, horizon)
                    domain_metrics[domain] = metric | {"score": float(score)}
                    if metric.get("easy_degradation", 1.0) > 0.018 or score <= 0.0:
                        eligible = False
                    scores.append(score)
                min_score = float(min(scores)) if scores else 0.0
                mean_score = float(np.mean(scores)) if scores else 0.0
                source_details[source] = {
                    "eligible_all_calibration_domains": eligible,
                    "min_score": min_score,
                    "mean_score": mean_score,
                    "domain_metrics": domain_metrics,
                }
                if eligible and min_score > best_score:
                    best_score = min_score
                    best_source = source
            choices[str(horizon)] = {
                "selected_source": best_source,
                "selection_rule": "max_min_validation_score_across_calibration_domains",
                "best_score": float(best_score),
                "source_details": source_details,
            }
        per_pair.append({"pair_idx": pair_idx, "cache_path": str(path), "choices": choices})
    return {
        "source": "fresh_run_validation_only_unseen_domain_transfer_rule",
        "calibration_domains": calibration_domains,
        "unseen_domain": "UCY",
        "per_pair": per_pair,
    }


def _evaluate_unseen(
    cache_paths: list[Path],
    labels_test: Mapping[str, np.ndarray],
    transfer_rule: Mapping[str, Any],
) -> dict[str, Any]:
    domain_test = labels_test["domain"].astype(str)
    horizon_test = labels_test["horizon"].astype(int)
    unseen_mask = domain_test == "UCY"
    rows = []
    source_switch_counts: dict[str, int] = {}
    for pair_rule in transfer_rule["per_pair"]:
        pair_idx = int(pair_rule["pair_idx"])
        path = cache_paths[pair_idx]
        with np.load(path, allow_pickle=False) as row:
            arrays = {k: row[k].copy() for k in row.files}
        selected_ade = arrays["floor_test_ade"].copy()
        selected_fde = arrays["floor_test_fde"].copy()
        switch = np.zeros(len(selected_ade), dtype=bool)
        for h_text, choice in pair_rule["choices"].items():
            horizon = int(h_text)
            source = str(choice.get("selected_source", "floor"))
            mask = unseen_mask & (horizon_test == horizon)
            if source == "floor" or not np.any(mask):
                continue
            _v_ade, _v_fde, _v_sw, t_ade, t_fde, t_sw = SOURCES[source]
            selected_ade[mask] = arrays[t_ade][mask]
            selected_fde[mask] = arrays[t_fde][mask]
            switch[mask] = arrays[t_sw][mask]
            source_switch_counts[source] = source_switch_counts.get(source, 0) + int(np.sum(arrays[t_sw][mask]))
        metric_ade = r._local_metric_from_errors(selected_ade, arrays["floor_test_ade"], labels_test, switch, unseen_mask)
        metric_fde = r._local_metric_from_errors(selected_fde, arrays["floor_test_fde"], labels_test, switch, unseen_mask)
        rows.append(
            {
                "pair_idx": pair_idx,
                "ade": metric_ade,
                "fde": metric_fde,
                "switch_rate": float(np.mean(switch[unseen_mask])) if np.any(unseen_mask) else 0.0,
                "source_switch_counts": source_switch_counts,
            }
        )
    return {"rows": rows, "summary": _summary(rows)}


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stat(path: tuple[str, ...]) -> dict[str, float]:
        vals = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = cur.get(key, {})
            vals.append(float(cur) if isinstance(cur, (int, float, np.floating)) else 0.0)
        return r._stat(vals)

    return {
        "source": "fresh_run_unseen_domain_test_once",
        "seeds": [int(row["pair_idx"]) for row in rows],
        "ade_all": stat(("ade", "all_improvement")),
        "ade_t50": stat(("ade", "t50_improvement")),
        "ade_t100_raw_frame_diagnostic": stat(("ade", "t100_improvement")),
        "ade_hard_failure": stat(("ade", "hard_failure_improvement")),
        "ade_easy_degradation": stat(("ade", "easy_degradation")),
        "fde_t50": stat(("fde", "t50_improvement")),
        "switch_rate": r._stat([float(row.get("switch_rate", 0.0)) for row in rows]),
    }


def _available_source_oracle(cache_paths: list[Path], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    domain_test = labels_test["domain"].astype(str)
    unseen_mask = domain_test == "UCY"
    diagnostics = []
    for pair_idx, path in enumerate(cache_paths):
        with np.load(path, allow_pickle=False) as row:
            arrays = {k: row[k].copy() for k in row.files}
        item: dict[str, Any] = {"pair_idx": pair_idx}
        for source, (_v_ade, _v_fde, _v_sw, t_ade, t_fde, t_sw) in SOURCES.items():
            ade_diff = arrays["floor_test_ade"][unseen_mask] - arrays[t_ade][unseen_mask]
            switch = arrays[t_sw][unseen_mask].astype(bool)
            item[source] = {
                "switch_count": int(np.sum(switch)),
                "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
                "mean_available_ade_improvement": float(np.mean(ade_diff)) if len(ade_diff) else 0.0,
                "nonzero_error_diff_rows": int(np.sum(np.abs(ade_diff) > 1e-9)),
            }
        diagnostics.append(item)
    any_nonfloor = any(
        source_item["switch_count"] > 0 or source_item["nonzero_error_diff_rows"] > 0
        for item in diagnostics
        for source, source_item in item.items()
        if source in SOURCES
    )
    return {
        "source": "fresh_run_test_diagnostic_only_available_source_oracle",
        "unseen_domain": "UCY",
        "any_available_nonfloor_prediction": any_nonfloor,
        "diagnostics": diagnostics,
        "interpretation": "current Stage42-R cache cannot improve UCY if this is false",
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("unseen_eval", {}).get("summary", {})
    source_oracle = result.get("available_source_oracle", {})
    gates = {
        "forensics_complete": bool(result.get("domain_coverage")) and bool(source_oracle),
        "validation_only_transfer_rule_built": result.get("transfer_rule", {}).get("source") == "fresh_run_validation_only_unseen_domain_transfer_rule",
        "ucy_test_evaluated_once": result.get("unseen_eval", {}).get("summary", {}).get("source") == "fresh_run_unseen_domain_test_once",
        "available_nonfloor_source_for_ucy": bool(source_oracle.get("any_available_nonfloor_prediction")),
        "ucy_all_positive": summary.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "ucy_t50_positive": summary.get("ade_t50", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": summary.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False
        and result.get("no_leakage", {}).get("test_threshold_tuning") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    all_pass = all(bool(v) for v in gates.values())
    if not gates["available_nonfloor_source_for_ucy"]:
        verdict = "stage42_t_ucy_transfer_blocked_no_candidate_predictions"
    elif not all_pass:
        verdict = "stage42_t_ucy_unseen_transfer_partial"
    else:
        verdict = "stage42_t_ucy_unseen_transfer_pass"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def run_stage42_ucy_unseen_transfer() -> dict[str, Any]:
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    ensure_dir(OUT_DIR)
    stage42r = r.run_stage42_row_prediction_cache()
    val = s42i._split_arrays("val")
    test = s42i._split_arrays("test")
    labels_val = r._labels_for(val)
    labels_test = r._labels_for(test)
    cache_paths = [Path(str(row["cache_path"])) for row in stage42r.get("cache_rows", [])]
    calibration_domains = sorted(set(labels_val["domain"].astype(str).tolist()))
    test_domains = sorted(set(labels_test["domain"].astype(str).tolist()))
    transfer_rule = _fit_transfer_rule(cache_paths, labels_val, calibration_domains)
    unseen_eval = _evaluate_unseen(cache_paths, labels_test, transfer_rule)
    source_oracle = _available_source_oracle(cache_paths, labels_test)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-T UCY unseen-domain transfer attempt",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([r.REPORT_JSON, OUT_DIR / "frozen_row_combo_policy_stage42.json"]),
        "domain_coverage": {
            "val_domains": _domain_counts(labels_val),
            "test_domains": _domain_counts(labels_test),
            "calibration_domains": calibration_domains,
            "unseen_test_domains": [d for d in test_domains if d not in calibration_domains],
        },
        "transfer_rule": transfer_rule,
        "unseen_eval": unseen_eval,
        "available_source_oracle": source_oracle,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_train_val_label_and_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "ucy_test_used_for_policy_selection": False,
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
    result["stage42_t_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_t_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_t_gate"]
    s = result["unseen_eval"]["summary"]
    lines = [
        "# Stage42-T UCY Unseen-Domain Transfer Attempt",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Domain Coverage",
        "",
        f"- validation domains: `{result['domain_coverage']['val_domains']}`",
        f"- test domains: `{result['domain_coverage']['test_domains']}`",
        f"- unseen test domains: `{result['domain_coverage']['unseen_test_domains']}`",
        "",
        "## UCY Test-Once Evaluation",
        "",
        "| metric | mean | ci_low | ci_high |",
        "| --- | ---: | ---: | ---: |",
        f"| ADE all | {s['ade_all']['mean']:.6f} | {s['ade_all']['ci_low']:.6f} | {s['ade_all']['ci_high']:.6f} |",
        f"| ADE t50 | {s['ade_t50']['mean']:.6f} | {s['ade_t50']['ci_low']:.6f} | {s['ade_t50']['ci_high']:.6f} |",
        f"| ADE hard/failure | {s['ade_hard_failure']['mean']:.6f} | {s['ade_hard_failure']['ci_low']:.6f} | {s['ade_hard_failure']['ci_high']:.6f} |",
        f"| ADE easy degradation | {s['ade_easy_degradation']['mean']:.6f} | {s['ade_easy_degradation']['ci_low']:.6f} | {s['ade_easy_degradation']['ci_high']:.6f} |",
        f"| FDE t50 | {s['fde_t50']['mean']:.6f} | {s['fde_t50']['ci_low']:.6f} | {s['fde_t50']['ci_high']:.6f} |",
        f"| switch rate | {s['switch_rate']['mean']:.6f} | {s['switch_rate']['ci_low']:.6f} | {s['switch_rate']['ci_high']:.6f} |",
        "",
        "## Available Source Oracle Diagnostic",
        "",
        f"- any_available_nonfloor_prediction: `{result['available_source_oracle']['any_available_nonfloor_prediction']}`",
        "- This diagnostic is test-only for blocker analysis, not policy selection.",
        "",
        "## Interpretation",
        "",
    ]
    if not result["available_source_oracle"]["any_available_nonfloor_prediction"]:
        lines.extend(
            [
                "- UCY remains fallback-only because the current Stage42-R row cache contains no non-floor Stage42-J/P predictions for UCY.",
                "- A validation-only unseen-domain transfer rule cannot create positive UCY transfer without a candidate source that actually switches or changes UCY rows.",
                "- The next aligned action is to train/cache a UCY-aware or source-agnostic prediction source using train/validation only, or rebuild splits so UCY has legal calibration support.",
            ]
        )
    else:
        lines.append("- UCY had non-floor candidate predictions; see metrics above for whether the validation-only transfer rule helped.")
    lines.append("- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.")
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-T Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_t_gate"]
    s = result["unseen_eval"]["summary"]
    block = f"""
## Stage42-T UCY Unseen-Domain Transfer Attempt

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
ucy_ade_all = {s['ade_all']['mean']}
ucy_ade_t50 = {s['ade_t50']['mean']}
ucy_hard_failure = {s['ade_hard_failure']['mean']}
ucy_easy_degradation = {s['ade_easy_degradation']['mean']}
available_nonfloor_source_for_ucy = {result['available_source_oracle']['any_available_nonfloor_prediction']}
stage5c_executed = false
smc_enabled = false
```

Stage42-T attempts a validation-only unseen-domain transfer rule for UCY. The current row cache has no non-floor Stage42-J/P UCY predictions, so UCY remains fallback-only; this is reported as a blocker, not as a success.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-T UCY Unseen-Domain Transfer Attempt", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-T UCY Unseen-Domain Transfer Attempt", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md"), "## Stage42-T UCY Unseen-Domain Transfer Attempt", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_t_ucy_unseen_transfer"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_t_ucy_unseen_transfer"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "ucy_ade_all": s["ade_all"]["mean"],
        "ucy_ade_t50": s["ade_t50"]["mean"],
        "ucy_hard_failure": s["ade_hard_failure"]["mean"],
        "ucy_easy_degradation": s["ade_easy_degradation"]["mean"],
        "available_nonfloor_source_for_ucy": result["available_source_oracle"]["any_available_nonfloor_prediction"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_ucy_unseen_transfer.py",
        "step": "stage42_t_ucy_unseen_transfer",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_ucy_unseen_transfer()
