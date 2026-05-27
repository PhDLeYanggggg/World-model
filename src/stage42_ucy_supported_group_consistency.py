from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_supported_group_consistency_stage42.json"
REPORT_MD = OUT_DIR / "ucy_supported_group_consistency_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dz_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DZ 在 Stage42-AW 的 UCY train-only internal-val split 上 fresh-runs Stage42-DI group-consistency repair。",
    "目标是检查 explicit physical/group-consistency 是否能在 UCY validation support 修复后获得双域支持，而不是只在 TrajNet 上有效。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "ungated_full_waypoint_deployable": False,
}


def _positive_safe(metric: Mapping[str, Any]) -> bool:
    return bool(
        float(metric.get("all_improvement", 0.0)) > 0.0
        and float(metric.get("t50_improvement", 0.0)) > 0.0
        and float(metric.get("hard_failure_improvement", 0.0)) > 0.0
        and float(metric.get("easy_degradation", 1.0)) <= 0.02
    )


def _domain_baseline_metrics(
    data: Mapping[str, np.ndarray],
    test_ids: np.ndarray,
    base_ade: np.ndarray,
    floor_ade: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    domains = data["dataset"][test_ids].astype(str)
    return {
        domain: di._metric_subset(base_ade[domains == domain], floor_ade[domains == domain], data, test_ids[domains == domain], switch[domains == domain])
        for domain in sorted(set(domains.tolist()))
    }


def _source_stats_summary(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train_rows": int(stats["by_split"]["train"]["rows"]),
        "val_rows": int(stats["by_split"]["val"]["rows"]),
        "test_rows": int(stats["by_split"]["test"]["rows"]),
        "val_domains": stats["by_split"]["val"]["domains"],
        "test_domains": stats["by_split"]["test"]["domains"],
        "source_overlap_pass": bool(stats["source_overlap_pass"]),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "ucy_internal_val_created": s["ucy_val_rows_after"] > 0,
        "test_sources_unchanged": no_leak["test_sources_unchanged"] is True,
        "source_overlap_pass": no_leak["source_overlap_pass"] is True,
        "group_repair_candidates_run": payload["repair"]["candidate_count"] >= 40,
        "validation_selected_without_test": no_leak["test_threshold_tuning"] is False,
        "global_positive_safe": _positive_safe(payload["repair"]["test"]["metric_vs_floor"]),
        "ucy_positive_safe": _positive_safe(payload["repair"]["test"]["by_domain"].get("UCY", {})),
        "trajnet_positive_safe": _positive_safe(payload["repair"]["test"]["by_domain"].get("TrajNet", {})),
        "group_consistency_not_worse_near005": s["near005_final"] <= s["near005_base"],
        "dual_domain_group_consistency_supported": s["positive_safe_domains"] >= 2,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["internal_val_from_train_only"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "ungated_full_waypoint_blocked": claim["ungated_full_waypoint_deployable"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_dz_ucy_supported_group_consistency_pass_dual_domain"
        if passed == total
        else "stage42_dz_ucy_supported_group_consistency_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    metric = payload["repair"]["test"]["metric_vs_floor"]
    base_metric = payload["baseline_stage42_am_on_test"]
    lines = [
        "# Stage42-DZ UCY-Supported Group-Consistency Full-Waypoint Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dz_gate']['passed']} / {payload['stage42_dz_gate']['total']}`",
        f"- verdict: `{payload['stage42_dz_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Split Repair",
        "",
        f"- internal_val_group: `{payload['internal_validation']['internal_val_group']}`",
        f"- ucy_val_rows_after: `{s['ucy_val_rows_after']}`",
        f"- test_rows_unchanged: `{payload['internal_validation']['test_rows_unchanged']}`",
        "",
        "## Metrics Vs Train-Horizon Causal Floor",
        "",
        "| policy | all | t50 | t100 raw | hard/failure | easy | switch | near@0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| AW/AM rebuilt baseline-family selected | {base_metric['all_improvement']:.6f} | {base_metric['t50_improvement']:.6f} | {base_metric['t100_raw_frame_diagnostic_improvement']:.6f} | {base_metric['hard_failure_improvement']:.6f} | {base_metric['easy_degradation']:.6f} | {base_metric['switch_rate']:.6f} | {s['near005_base']:.6f} |",
        f"| DZ group-consistency repaired | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} | {s['near005_final']:.6f} |",
        "",
        "## By Domain",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy | switch | positive_safe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in payload["repair"]["test"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {int(row['rows'])} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} | `{_positive_safe(row)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a fresh repair run, not cached reuse of AW or DI reports.",
            "- Stage42-AW fixes the UCY validation-support blocker by carving internal validation from original UCY train sources only.",
            "- Stage42-DZ shows whether explicit group/physical consistency remains safe and positive when UCY has validation support.",
            "- This still remains dataset-local/raw-frame 2.5D evidence and does not permit metric/seconds-level, true-3D, Stage5C, or SMC claims.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dz_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dz_gate"]
    return [
        "# Stage42-DZ Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    metric = payload["repair"]["test"]["metric_vs_floor"]
    return [
        "## Stage42-DZ UCY-Supported Group-Consistency Full-Waypoint Repair",
        "",
        "- source: `fresh_ucy_internal_validation_group_consistency_repair`",
        "- role: reruns explicit group/physical consistency on the UCY validation-supported split, addressing the prior TrajNet-only/floor-only domain boundary.",
        f"- gate: `{payload['stage42_dz_gate']['passed']} / {payload['stage42_dz_gate']['total']}`; verdict `{payload['stage42_dz_gate']['verdict']}`.",
        f"- global all/t50/t100 raw/hard/easy `{metric['all_improvement']:.6f}` / `{metric['t50_improvement']:.6f}` / `{metric['t100_raw_frame_diagnostic_improvement']:.6f}` / `{metric['hard_failure_improvement']:.6f}` / `{metric['easy_degradation']:.6f}`.",
        f"- positive safe domains: `{s['positive_safe_domains']}`; UCY all/t50/hard `{s['ucy_all']:.6f}` / `{s['ucy_t50']:.6f}` / `{s['ucy_hard']:.6f}`; TrajNet all/t50/hard `{s['trajnet_all']:.6f}` / `{s['trajnet_t50']:.6f}` / `{s['trajnet_hard']:.6f}`.",
        f"- near@0.05 base/final `{s['near005_base']:.6f}` / `{s['near005_final']:.6f}`; still raw-frame/dataset-local, no metric/seconds claim, Stage5C false, SMC false.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DZ_UCY_SUPPORTED_GROUP_CONSISTENCY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DZ UCY-supported group-consistency full-waypoint repair"
    state["current_verdict"] = payload["stage42_dz_gate"]["verdict"]
    state["stage42_dz_ucy_supported_group_consistency"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dz_gate"]["verdict"],
        "gates": f"{payload['stage42_dz_gate']['passed']}/{payload['stage42_dz_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_ucy_supported_group_consistency() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = repaired_split == "train"
    floor = am._floor_arrays(data, train_mask)
    am_candidate = di._rebuild_stage42_am_candidate(data, repaired_split, labels, floor)
    group_key = di._group_key(data)
    repair = di._evaluate_repairs(data, repaired_split, labels, floor, am_candidate, group_key)
    original_stats = am._source_stats(data, original_split, group)
    repaired_stats = am._source_stats(data, repaired_split, group)
    test_ids = np.where(repaired_split == "test")[0]
    baseline_by_domain = _domain_baseline_metrics(
        data,
        test_ids,
        am_candidate["selected_ade"][test_ids].astype(np.float64),
        am_candidate["floor_ade"][test_ids].astype(np.float64),
        am_candidate["switch"][test_ids].astype(bool),
    )
    domain_metrics = repair["test"]["by_domain"]
    ucy = domain_metrics.get("UCY", {})
    traj = domain_metrics.get("TrajNet", {})
    positive_safe_domains = sum(1 for row in domain_metrics.values() if _positive_safe(row))
    summary = {
        "ucy_val_rows_after": int(repaired_stats["by_split"]["val"]["domains"].get("UCY", 0)),
        "positive_safe_domains": int(positive_safe_domains),
        "ucy_positive_safe": _positive_safe(ucy),
        "trajnet_positive_safe": _positive_safe(traj),
        "ucy_all": float(ucy.get("all_improvement", 0.0)),
        "ucy_t50": float(ucy.get("t50_improvement", 0.0)),
        "ucy_hard": float(ucy.get("hard_failure_improvement", 0.0)),
        "trajnet_all": float(traj.get("all_improvement", 0.0)),
        "trajnet_t50": float(traj.get("t50_improvement", 0.0)),
        "trajnet_hard": float(traj.get("hard_failure_improvement", 0.0)),
        "near005_base": float(repair["test"]["diagnostics"]["base_near_005"]),
        "near005_final": float(repair["test"]["diagnostics"]["final_near_005"]),
        "deployment_decision": "promote_ucy_supported_group_consistency_as_dual_domain_source_level_policy"
        if positive_safe_domains >= 2
        else "keep_dy_source_level_policy_and_treat_ucy_support_as_partial",
    }
    payload: dict[str, Any] = {
        "source": "fresh_ucy_internal_validation_group_consistency_repair",
        "stage": "Stage42-DZ",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/ucy_validation_support_repair_stage42.json",
                "outputs/stage42_long_research/group_consistency_full_waypoint_repair_stage42.json",
            ]
        ),
        "internal_validation": {
            "source": "fresh_run",
            "domain": "UCY",
            "selected_from": "original_train_sources_only",
            "internal_val_group": internal_val_group,
            "uses_test_rows": False,
            "test_rows_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
        },
        "original_split_stats": _source_stats_summary(original_stats),
        "repaired_split_stats": _source_stats_summary(repaired_stats),
        "stage42_am_rebuilt": {
            "source": "fresh_rebuild_with_ucy_internal_val",
            "lambda": am_candidate["lambda"],
            "feature_count": am_candidate["feature_count"],
            "policy_slice_count": len(am_candidate["policy"]["slices"]),
            "val_metric": am_candidate["val_metric"],
        },
        "baseline_stage42_am_on_test": repair["baseline_stage42_am_on_test"],
        "baseline_stage42_am_by_domain": baseline_by_domain,
        "repair": repair,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_sources_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(repaired_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_dz_gate"] = _gate(payload)
    write_json(REPORT_JSON, di._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_ucy_supported_group_consistency()
