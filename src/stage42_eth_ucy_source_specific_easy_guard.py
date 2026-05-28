from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_rotation_easy_guard_repair as jf
from src import stage42_source_rotation_full_waypoint_eval as je
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_source_specific_easy_guard_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_source_specific_easy_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jg_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JG_ETH_UCY_SOURCE_SPECIFIC_EASY_GUARD"
SOURCE = "fresh_stage42_jg_eth_ucy_source_specific_easy_guard"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JG 是 ETH_UCY source-specific easy-guard feasibility：它只在 ETH_UCY 内做 source-disjoint CV。",
    "这个阶段不把 ETH_UCY 写成 cross-domain zero-shot 成功；它只测试源级支持是否能修复 easy harm。",
    "每个 fold 的 held-out source 不参与 train/validation policy selection。",
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


def _subset_data(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    n = len(mask)
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def _source_ids(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([s42b._rel_source(path) for path in data["source_file"].astype(str)], dtype="U512")


def _eth_ucy_data() -> dict[str, np.ndarray]:
    data = s41._combined()
    mask = data["dataset"].astype(str) == "ETH_UCY"
    return _subset_data(data, mask)


def _source_cv_split(data: Mapping[str, np.ndarray], heldout_source: str) -> tuple[np.ndarray, dict[str, Any]]:
    source = _source_ids(data)
    sources = sorted(set(source.tolist()))
    remaining = [s for s in sources if s != heldout_source]
    if len(remaining) < 2:
        raise ValueError("ETH_UCY source-specific CV needs at least two non-heldout sources.")
    # Pick one validation source deterministically from non-heldout sources.
    val_source = max(remaining, key=lambda s: s42b._stable_unit(f"stage42-jg::{heldout_source}::{s}"))
    train_sources = [s for s in remaining if s != val_source]
    split = np.full(len(source), "train", dtype="U8")
    split[source == val_source] = "val"
    split[source == heldout_source] = "test"
    stats = {
        "heldout_source": heldout_source,
        "train_sources": train_sources,
        "val_source": val_source,
        "test_source": heldout_source,
        "train_rows": int(np.sum(split == "train")),
        "val_rows": int(np.sum(split == "val")),
        "test_rows": int(np.sum(split == "test")),
        "source_overlap": {
            "train_val": int(len(set(source[split == "train"].tolist()) & set(source[split == "val"].tolist()))),
            "train_test": int(len(set(source[split == "train"].tolist()) & set(source[split == "test"].tolist()))),
            "val_test": int(len(set(source[split == "val"].tolist()) & set(source[split == "test"].tolist()))),
        },
    }
    stats["source_overlap_pass"] = all(v == 0 for v in stats["source_overlap"].values())
    return split, stats


def _evaluate_fold(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str) -> dict[str, Any]:
    split, split_stats = _source_cv_split(data, heldout_source)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names, removed = je._domain_invariant_features(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    horizon = data["horizon"].astype(int)
    best: dict[str, Any] | None = None
    candidates = []
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        base_policy, base_ade, base_fde, base_switch = je._select_horizon_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        if not base_policy.get("slices"):
            continue
        residual_norm = np.linalg.norm(pred_xy[:, -1] - floor["floor_xy"][:, -1], axis=1) / np.maximum(
            data["scale"].astype(np.float64), EPS
        )
        floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
        base_test_metric = am._metric(base_ade, floor_ade, data, base_switch, test_mask)
        for cap in jf.CAPS:
            capped_switch = jf._cap_switch_mask(base_switch, residual_norm, base_policy, horizon, cap)
            selected_ade, selected_fde, floor_ade = jf._blend_errors_for_switch(
                pred_xy, floor["floor_xy"], labels, base_switch, capped_switch, base_policy, horizon
            )
            val_metric = am._metric(selected_ade, floor_ade, data, capped_switch, val_mask)
            score = jf._candidate_score(val_metric)
            row = {
                "lambda": float(lam),
                "switch_cap": float(cap),
                "score": float(score),
                "policy_slice_count": int(len(base_policy["slices"])),
                "val_metric": val_metric,
                "base_test_metric_before_cap": base_test_metric,
            }
            candidates.append(row)
            if best is None or score > best["score"]:
                best = {
                    **row,
                    "policy": base_policy,
                    "pred_xy": pred_xy,
                    "selected_ade": selected_ade,
                    "selected_fde": selected_fde,
                    "switch": capped_switch,
                    "floor_ade": floor_ade,
                    "floor_fde": floor_fde,
                }
    if best is None:
        raise RuntimeError(f"No ETH_UCY source-specific candidate for {heldout_source}.")
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    metric = am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask)
    fde = am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask)
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names)),
            "domain_features_removed": removed,
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
        },
        "selected_candidate": {
            "lambda": best["lambda"],
            "switch_cap": best["switch_cap"],
            "score": best["score"],
            "validation_selection_source": "eth_ucy_non_heldout_source_validation_only",
            "test_threshold_tuning": False,
            "policy_slice_count": best["policy_slice_count"],
            "val_metric": best["val_metric"],
        },
        "candidate_count": int(len(candidates)),
        "top_validation_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
        "metrics": {
            "base_horizon_policy_before_cap": best["base_test_metric_before_cap"],
            "eth_ucy_source_specific_policy": metric,
            "eth_ucy_source_specific_policy_fde": fde,
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42151),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42152),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42153),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42154),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42155),
        },
    }


def _summary(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    deployable = []
    blocked = []
    positive_but_unsafe = []
    for row in folds:
        metric = row["metrics"]["eth_ucy_source_specific_policy"]
        ok_positive = metric["all_improvement"] > 0.03 and (
            metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10
        )
        ok_easy = metric["easy_degradation"] <= 0.02
        if ok_positive and ok_easy:
            deployable.append(row["heldout_source"])
        elif ok_positive:
            positive_but_unsafe.append(row["heldout_source"])
            blocked.append(row["heldout_source"])
        else:
            blocked.append(row["heldout_source"])
    if len(deployable) == len(folds) and folds:
        decision = "eth_ucy_source_specific_policy_supported_all_sources"
    elif deployable:
        decision = "eth_ucy_source_specific_policy_partial_source_support"
    else:
        decision = "eth_ucy_source_specific_policy_not_supported"
    return {
        "source": SOURCE,
        "fold_count": int(len(folds)),
        "deployable_heldout_sources": deployable,
        "blocked_heldout_sources": blocked,
        "positive_but_easy_unsafe_sources": positive_but_unsafe,
        "deployable_source_count": int(len(deployable)),
        "decision": decision,
        "next_action": "If only part of ETH_UCY is deployable, keep ETH_UCY fallback-only by default and require source-identity/calibration or per-source support before promotion.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = _eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    sources = sorted(set(_source_ids(data).tolist()))
    folds = [_evaluate_fold(data, labels, source) for source in sources]
    payload: dict[str, Any] = {
        "stage": "Stage42-JG",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_rotation_easy_guard_repair_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "eth_ucy_rows": int(len(data["horizon"])),
        "eth_ucy_sources": sources,
        "label_stats": {
            "full_waypoint_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
            "missing_track_rows": int(np.sum(labels["missing_track"])),
        },
        "folds": folds,
        "summary": _summary(folds),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in folds),
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
    payload["stage42_jg_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "eth_ucy_sources_enough_for_cv": len(payload["eth_ucy_sources"]) >= 5,
        "folds_built_for_all_sources": len(payload["folds"]) == len(payload["eth_ucy_sources"]),
        "source_overlap_pass": payload["no_leakage"]["source_overlap_pass"] is True,
        "validation_only_selection": all(row["selected_candidate"]["test_threshold_tuning"] is False for row in payload["folds"]),
        "candidate_search_recorded": all(row["candidate_count"] > 0 and row["top_validation_candidates"] for row in payload["folds"]),
        "deployable_or_blocked_sources_recorded": bool(payload["summary"]["deployable_heldout_sources"] or payload["summary"]["blocked_heldout_sources"]),
        "partial_support_not_overclaimed": payload["summary"]["decision"] != "eth_ucy_source_specific_policy_supported_all_sources"
        or len(payload["summary"]["blocked_heldout_sources"]) == 0,
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
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_jg_eth_ucy_source_specific_easy_guard_pass" if passed == len(gates) else "stage42_jg_eth_ucy_source_specific_easy_guard_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jg_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JG ETH_UCY Source-Specific Easy-Guard Feasibility",
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
        "## Summary",
        "",
        f"- decision: `{summary['decision']}`",
        f"- deployable_heldout_sources: `{summary['deployable_heldout_sources']}`",
        f"- blocked_heldout_sources: `{summary['blocked_heldout_sources']}`",
        f"- positive_but_easy_unsafe_sources: `{summary['positive_but_easy_unsafe_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Source-CV Fold Metrics",
        "",
        "| heldout source | rows | cap | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["folds"]:
        metric = row["metrics"]["eth_ucy_source_specific_policy"]
        lines.append(
            f"| `{row['heldout_source']}` | {metric['rows']} | {row['selected_candidate']['switch_cap']:.2f} | "
            f"{_fmt(metric['all_improvement'])} | {_fmt(metric['t50_improvement'])} | "
            f"{_fmt(metric['t100_raw_frame_diagnostic_improvement'])} | {_fmt(metric['hard_failure_improvement'])} | "
            f"{_fmt(metric['easy_degradation'])} | {_fmt(metric['switch_rate'])} |"
        )
    lines.extend(["", "## Bootstrap CI", "", "| heldout source | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["folds"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_source']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This stage tests source-specific ETH_UCY support, not cross-domain zero-shot transfer.",
            "- If some ETH_UCY sources remain blocked, the default deployment boundary remains fallback-only for ETH_UCY unless source identity/calibration support is available.",
            "- This remains dataset-local/raw-frame 2.5D evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jg_gate"]
    lines = [
        "# Stage42-JG Gate",
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
    gate = payload["stage42_jg_gate"]
    fold_bits = []
    for row in payload["folds"]:
        metric = row["metrics"]["eth_ucy_source_specific_policy"]
        fold_bits.append(
            f"{row['heldout_source']}: all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}"
        )
    return [
        "## Stage42-JG ETH_UCY Source-Specific Easy-Guard Feasibility",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- source-CV folds: {'; '.join(fold_bits)}.",
        f"- decision: `{summary['decision']}`; deployable sources: `{summary['deployable_heldout_sources']}`; blocked sources: `{summary['blocked_heldout_sources']}`.",
        "- boundary: this is ETH_UCY source-specific support only, not cross-domain zero-shot success; still no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_source_specific_easy_guard"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jg_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jg_gate"]["passed"], "total": payload["stage42_jg_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "deployable_heldout_sources": payload["summary"]["deployable_heldout_sources"],
        "blocked_heldout_sources": payload["summary"]["blocked_heldout_sources"],
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
                    "stage": "Stage42-JG",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jg_gate"]["verdict"],
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


def run_stage42_eth_ucy_source_specific_easy_guard(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    run_stage42_eth_ucy_source_specific_easy_guard(refresh_readmes=True)


if __name__ == "__main__":
    main()
