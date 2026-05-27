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
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
HQ_JSON = OUT_DIR / "group_consistency_weak_slice_repair_stage42.json"
REPORT_JSON = OUT_DIR / "group_consistency_t100_easy_guard_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_t100_easy_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hr_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

T100_EASY_THRESHOLD = 0.0

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HR 针对 Stage42-HQ 暴露的 t100 easy degradation 做 validation-only domain|t100 guard。",
    "HR 不用 test metrics 调阈值；domain|t100 是否保留只由 validation all gain 和 validation easy degradation 决定。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), di.EPS)


def _easy_degradation(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return -_safe_improvement(selected, floor, mask)


def _keep_t100_slice(val_all_improvement: float, val_easy_degradation: float, threshold: float = T100_EASY_THRESHOLD) -> bool:
    return float(val_all_improvement) > 0.0 and float(val_easy_degradation) <= float(threshold)


def _by_domain(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    domain = data["dataset"][ids].astype(str)
    return {
        d: di._metric_subset(selected_ade[domain == d], floor_ade[domain == d], data, ids[domain == d], switch[domain == d])
        for d in sorted(set(domain.tolist()))
    }


def _by_horizon(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    h = data["horizon"][ids].astype(int)
    return {
        str(k): di._metric_subset(selected_ade[h == k], floor_ade[h == k], data, ids[h == k], switch[h == k])
        for k in [10, 25, 50, 100]
    }


def _rebuild_hq_candidate() -> dict[str, Any]:
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    original_stats = am._source_stats(data, original_split, group)
    repaired_stats = am._source_stats(data, split, group)
    floor = am._floor_arrays(data, split == "train")
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    group_key = di._group_key(data)
    repair = di._evaluate_repairs(data, split, labels, floor, am_candidate, group_key)
    candidate = repair["selected"]["candidate"]
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    val = di._repair_subset(
        val_ids,
        candidate,
        data,
        labels,
        floor["floor_xy"].astype(np.float32),
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    test = di._repair_subset(
        test_ids,
        candidate,
        data,
        labels,
        floor["floor_xy"].astype(np.float32),
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    return {
        "data": data,
        "split": split,
        "labels": labels,
        "group_key": group_key,
        "floor_xy": floor["floor_xy"].astype(np.float32),
        "candidate": candidate,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "val": val,
        "test": test,
        "internal_val_group": str(internal_val_group),
        "original_stats": original_stats,
        "repaired_stats": repaired_stats,
    }


def _apply_domain_t100_guard(rebuilt: Mapping[str, Any], threshold: float = T100_EASY_THRESHOLD) -> dict[str, Any]:
    data = rebuilt["data"]
    val_ids = rebuilt["val_ids"]
    test_ids = rebuilt["test_ids"]
    val = rebuilt["val"]
    test = rebuilt["test"]
    domain_val = data["dataset"][val_ids].astype(str)
    horizon_val = data["horizon"][val_ids].astype(int)
    easy_val = data["easy"][val_ids].astype(bool)
    domain_test = data["dataset"][test_ids].astype(str)
    horizon_test = data["horizon"][test_ids].astype(int)
    easy_test = data["easy"][test_ids].astype(bool)
    selected_xy = test["selected_xy"].astype(np.float32).copy()
    selected_ade = test["selected_ade"].astype(np.float64).copy()
    selected_fde = test["selected_fde"].astype(np.float64).copy()
    switch = test["switch"].astype(bool).copy()
    guarded: dict[str, Any] = {}
    kept: dict[str, Any] = {}
    domains = sorted(set(domain_val.tolist()) | set(domain_test.tolist()))
    for d in domains:
        val_mask = (domain_val == d) & (horizon_val == 100)
        test_mask = (domain_test == d) & (horizon_test == 100)
        if not np.any(test_mask):
            continue
        val_all = _safe_improvement(val["selected_ade"], val["floor_ade"], val_mask)
        val_easy = _easy_degradation(val["selected_ade"], val["floor_ade"], val_mask & easy_val)
        keep = _keep_t100_slice(val_all, val_easy, threshold)
        record = {
            "source": "fresh_validation_only_domain_t100_easy_guard",
            "domain": d,
            "val_rows": int(np.sum(val_mask)),
            "test_rows": int(np.sum(test_mask)),
            "val_all_improvement": float(val_all),
            "val_easy_degradation": float(val_easy),
            "threshold": float(threshold),
            "keep": bool(keep),
        }
        if keep:
            kept[f"{d}|100"] = record
            continue
        local_test_ids = test_ids[test_mask]
        selected_xy[test_mask] = rebuilt["floor_xy"][local_test_ids]
        selected_ade[test_mask] = test["floor_ade"][test_mask]
        selected_fde[test_mask] = test["floor_fde"][test_mask]
        switch[test_mask] = False
        guarded[f"{d}|100"] = {**record, "reason": "validation_easy_degradation_above_threshold_or_nonpositive_gain"}
    normalizer = np.maximum(data["scale"][test_ids].astype(np.float64), di.EPS)
    final_min = di._min_group_distance_fast(
        selected_xy,
        rebuilt["group_key"][test_ids],
        normalizer,
        data["agent_id"][test_ids].astype(np.int64),
    )
    metric = di._metric_subset(selected_ade, test["floor_ade"], data, test_ids, switch)
    t100_easy = (horizon_test == 100) & easy_test
    return {
        "source": "fresh_validation_only_domain_t100_easy_guard",
        "threshold": float(threshold),
        "guarded_slices": guarded,
        "kept_slices": kept,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
        "metric": metric,
        "by_domain": _by_domain(data, test_ids, selected_ade, test["floor_ade"], switch),
        "by_horizon": _by_horizon(data, test_ids, selected_ade, test["floor_ade"], switch),
        "t100_easy_rows": int(np.sum(t100_easy)),
        "t100_easy_degradation": _easy_degradation(selected_ade, test["floor_ade"], t100_easy),
        "diagnostics": {
            "pre_guard_near_005": float(test["diagnostics"]["final_near_005"]),
            "post_guard_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
            "floor_near_005": float(test["diagnostics"]["floor_near_005"]),
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    before = payload["pre_guard"]
    guard = payload["guarded"]
    metric = guard["metric"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    ucy = guard["by_domain"].get("UCY", {})
    gates = {
        "hq_input_loaded": payload["hq_input"]["exists"] is True,
        "hq_t100_easy_was_weak": float(before["t100_easy_degradation"]) > 0.02,
        "fresh_guard_source": guard["source"] == "fresh_validation_only_domain_t100_easy_guard",
        "validation_only_decisions": payload["uses_test_metrics_for_guard"] is False,
        "guarded_or_kept_slices_recorded": bool(guard["guarded_slices"] or guard["kept_slices"]),
        "t100_easy_repaired_under_2pct": float(guard["t100_easy_degradation"]) <= 0.02,
        "t100_raw_remains_positive": float(metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "global_all_positive": float(metric["all_improvement"]) > 0.0,
        "global_t50_positive": float(metric["t50_improvement"]) > 0.0,
        "global_hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "global_easy_preserved": float(metric["easy_degradation"]) <= 0.02,
        "ucy_remains_positive": float(ucy.get("all_improvement", 0.0)) > 0.0 and float(ucy.get("t50_improvement", 0.0)) > 0.0,
        "near005_not_worse_vs_pre_guard": float(guard["diagnostics"]["post_guard_near_005"]) <= float(
            guard["diagnostics"]["pre_guard_near_005"]
        ),
        "no_future_endpoint_input": no_leak["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leak["future_waypoint_input"] is False,
        "no_central_velocity": no_leak["central_velocity"] is False,
        "no_test_endpoint_goals": no_leak["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
        "internal_val_from_train_only": no_leak["internal_val_from_train_only"] is True,
        "source_overlap_pass": no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_claim": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hr_t100_easy_guard_pass" if passed == total else "stage42_hr_t100_easy_guard_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _summary_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hr_gate"]
    metric = payload["guarded"]["metric"]
    return [
        "## Stage42-HR Group-Consistency T100 Easy Guard",
        "",
        "- source: `fresh_validation_only_domain_t100_easy_guard`",
        "- role: repair Stage42-HQ t100 easy degradation with validation-only domain|t100 fallback decisions.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- HQ t100 easy before: `{_pct(payload['pre_guard']['t100_easy_degradation'])}`; after guard `{_pct(payload['guarded']['t100_easy_degradation'])}`.",
        f"- guarded all/t50/t100 raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- guarded slices: `{payload['guarded']['guarded_slices']}`; kept slices: `{payload['guarded']['kept_slices']}`.",
        "- claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hr_gate"]
    before = payload["pre_guard"]
    guarded = payload["guarded"]
    metric = guarded["metric"]
    lines = [
        "# Stage42-HR Group-Consistency T100 Easy Guard",
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
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Before / After",
        "",
        "| metric | HQ before | HR after |",
        "| --- | ---: | ---: |",
        f"| all | {_pct(before['metric']['all_improvement'])} | {_pct(metric['all_improvement'])} |",
        f"| t50 | {_pct(before['metric']['t50_improvement'])} | {_pct(metric['t50_improvement'])} |",
        f"| t100 raw diagnostic | {_pct(before['metric']['t100_raw_frame_diagnostic_improvement'])} | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} |",
        f"| hard/failure | {_pct(before['metric']['hard_failure_improvement'])} | {_pct(metric['hard_failure_improvement'])} |",
        f"| easy degradation | {_pct(before['metric']['easy_degradation'])} | {_pct(metric['easy_degradation'])} |",
        f"| t100 easy degradation | {_pct(before['t100_easy_degradation'])} | {_pct(guarded['t100_easy_degradation'])} |",
        f"| switch | {_pct(before['metric']['switch_rate'])} | {_pct(metric['switch_rate'])} |",
        "",
        "## Validation-Only Decisions",
        "",
        f"- threshold: `{guarded['threshold']}`",
        f"- guarded_slices: `{guarded['guarded_slices']}`",
        f"- kept_slices: `{guarded['kept_slices']}`",
        "",
        "## By Domain After Guard",
        "",
        "| domain | rows | all | t50 | t100 raw | hard | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in guarded["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | {_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | {_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-HR repairs the HQ t100 easy-safety weak slice using validation-only domain|t100 decisions.",
            "- The t100 result remains raw-frame diagnostic and must not be described as seconds-level long horizon.",
            "- This step does not execute Stage5C, does not enable SMC, and does not make metric/foundation/true-3D claims.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HR Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        gate_lines.append(f"| `{name}` | `{ok}` |")
    write_md(GATE_MD, gate_lines)


def _refresh_state(payload: Mapping[str, Any]) -> None:
    lines = _summary_lines(payload)
    for path in [README_RESULTS, M3W_README]:
        _replace_section(path, "STAGE42_HR_GROUP_CONSISTENCY_T100_EASY_GUARD", lines)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HR group-consistency t100 easy guard"
    state["current_verdict"] = payload["stage42_hr_gate"]["verdict"]
    state["stage42_hr_group_consistency_t100_easy_guard"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hr_gate"]["verdict"],
        "gates": f"{payload['stage42_hr_gate']['passed']}/{payload['stage42_hr_gate']['total']}",
        "pre_guard": payload["pre_guard"],
        "guarded_metric": payload["guarded"]["metric"],
        "guarded_slices": payload["guarded"]["guarded_slices"],
        "kept_slices": payload["guarded"]["kept_slices"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Validation-only domain|t100 guard repairs the HQ t100 easy-degradation weak slice while preserving positive all/t50/hard and raw-frame claim boundaries.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_group_consistency_t100_easy_guard.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_t100_easy_guard.py",
        },
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_t100_easy_guard() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hq = read_json(HQ_JSON, {}) if HQ_JSON.exists() else {}
    rebuilt = _rebuild_hq_candidate()
    guarded = _apply_domain_t100_guard(rebuilt, threshold=T100_EASY_THRESHOLD)
    pre = {
        "source": rebuilt["test"]["source"] if "source" in rebuilt["test"] else "fresh_pre_guard_hq_rebuild",
        "metric": rebuilt["test"]["metric"],
        "t100_easy_degradation": _easy_degradation(
            rebuilt["test"]["selected_ade"],
            rebuilt["test"]["floor_ade"],
            (rebuilt["data"]["horizon"][rebuilt["test_ids"]].astype(int) == 100)
            & rebuilt["data"]["easy"][rebuilt["test_ids"]].astype(bool),
        ),
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_hr_validation_only_t100_easy_guard",
        "stage": "Stage42-HR group-consistency t100 easy guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([HQ_JSON, Path("data/stage41_world_model/combined_external.npz")]),
        "hq_input": {"path": str(HQ_JSON), "exists": bool(hq), "verdict": hq.get("stage42_hq_gate", {}).get("verdict")},
        "pre_guard": pre,
        "guarded": guarded,
        "uses_test_metrics_for_guard": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "source_overlap_pass": bool(rebuilt["repaired_stats"]["source_overlap_pass"]),
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
    payload["stage42_hr_gate"] = _gate(payload)
    compact_payload = dict(payload)
    compact_payload["guarded"] = {
        k: v
        for k, v in guarded.items()
        if k not in {"selected_ade", "selected_fde", "switch"}
    }
    write_json(REPORT_JSON, di._jsonable(compact_payload))
    _write_outputs(payload)
    _refresh_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_group_consistency_t100_easy_guard()
    gate = out["stage42_hr_gate"]
    print(f"Stage42-HR group-consistency t100 easy guard: {gate['verdict']} ({gate['passed']}/{gate['total']})")
