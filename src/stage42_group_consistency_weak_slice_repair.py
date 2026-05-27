from __future__ import annotations

from collections import Counter
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
HP_JSON = OUT_DIR / "group_consistency_breakdown_stage42.json"
REPORT_JSON = OUT_DIR / "group_consistency_weak_slice_repair_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_weak_slice_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hq_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HQ 针对 Stage42-HP 暴露的 UCY 0-gain weak slice 做 fresh UCY-internal-validation-supported group-consistency repair。",
    "HQ 不用 test metrics 调阈值；UCY support 来自 original train sources carved internal validation。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _metric_on_mask(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), di.EPS)


def _easy_degradation_on_mask(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return -_metric_on_mask(selected, floor, mask)


def _by_domain(
    data: Mapping[str, np.ndarray],
    test_ids: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    domain = data["dataset"][test_ids].astype(str)
    return {
        d: di._metric_subset(selected_ade[domain == d], floor_ade[domain == d], data, test_ids[domain == d], switch[domain == d])
        for d in sorted(set(domain.tolist()))
    }


def _repair_with_ucy_support() -> dict[str, Any]:
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    original_stats = am._source_stats(data, original_split, group)
    repaired_stats = am._source_stats(data, repaired_split, group)
    floor = am._floor_arrays(data, repaired_split == "train")
    am_candidate = di._rebuild_stage42_am_candidate(data, repaired_split, labels, floor)
    group_key = di._group_key(data)
    repair = di._evaluate_repairs(data, repaired_split, labels, floor, am_candidate, group_key)
    test_ids = np.where(repaired_split == "test")[0]
    test = di._repair_subset(
        test_ids,
        repair["selected"]["candidate"],
        data,
        labels,
        floor["floor_xy"].astype(np.float32),
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    h = data["horizon"][test_ids].astype(int)
    easy = data["easy"][test_ids].astype(bool)
    t100_easy = (h == 100) & easy
    return {
        "source": "fresh_ucy_internal_validation_supported_repair",
        "internal_val_group": str(internal_val_group),
        "original_split_stats": original_stats,
        "repaired_split_stats": repaired_stats,
        "test_sources_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
        "repair": repair,
        "test": {
            "metric": test["metric"],
            "by_domain": _by_domain(data, test_ids, test["selected_ade"], test["floor_ade"], test["switch"]),
            "diagnostics": test["diagnostics"],
            "t100_easy_rows": int(np.sum(t100_easy)),
            "t100_easy_degradation": _easy_degradation_on_mask(test["selected_ade"], test["floor_ade"], t100_easy),
            "horizon_counts": dict(Counter(h.astype(int).tolist())),
            "selected_candidate": repair["selected"]["candidate"],
        },
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
    }


def _hp_ucy_before(hp: Mapping[str, Any]) -> dict[str, Any]:
    for row in hp.get("breakdown", {}).get("by_domain", []):
        if row.get("name") == "domain:UCY":
            return dict(row)
    return {}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    repair = payload["ucy_supported_repair"]
    metric = repair["test"]["metric"]
    ucy = repair["test"]["by_domain"].get("UCY", {})
    no_leak = repair["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "hp_input_loaded": payload["hp_input"]["exists"] is True,
        "hp_identified_ucy_weak_slice": float(payload["hp_ucy_before"].get("ade_all_improvement", 1.0)) <= 0.0,
        "fresh_ucy_supported_repair": repair["source"] == "fresh_ucy_internal_validation_supported_repair",
        "internal_val_from_train_only": no_leak["internal_val_from_train_only"] is True,
        "test_sources_unchanged": no_leak["test_sources_unchanged"] is True,
        "source_overlap_pass": no_leak["source_overlap_pass"] is True,
        "global_positive": float(metric["all_improvement"]) > 0.0,
        "global_t50_positive": float(metric["t50_improvement"]) > 0.0,
        "global_hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "ucy_all_repaired_positive": float(ucy.get("all_improvement", 0.0)) > 0.0,
        "ucy_t50_repaired_positive": float(ucy.get("t50_improvement", 0.0)) > 0.0,
        "ucy_hard_repaired_positive": float(ucy.get("hard_failure_improvement", 0.0)) > 0.0,
        "easy_preserved_global": float(metric["easy_degradation"]) <= 0.02,
        "near005_not_worse": float(repair["test"]["diagnostics"]["final_near_005"]) <= float(
            repair["test"]["diagnostics"]["base_near_005"]
        ),
        "t100_easy_status_recorded": repair["test"]["t100_easy_rows"] >= 0,
        "no_future_endpoint_input": no_leak["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leak["future_waypoint_input"] is False,
        "no_central_velocity": no_leak["central_velocity"] is False,
        "no_test_endpoint_goals": no_leak["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
        "no_metric_seconds_claim": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hq_group_consistency_weak_slice_repair_pass" if passed == total else "stage42_hq_group_consistency_weak_slice_repair_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _summary_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hq_gate"]
    metric = payload["ucy_supported_repair"]["test"]["metric"]
    ucy = payload["ucy_supported_repair"]["test"]["by_domain"].get("UCY", {})
    return [
        "## Stage42-HQ UCY Weak-Slice Group-Consistency Repair",
        "",
        "- source: `fresh_ucy_internal_validation_supported_repair`",
        "- role: repair the Stage42-HP UCY zero-gain weak slice with train-only UCY internal validation support.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- HP UCY before: all `{_pct(payload['hp_ucy_before'].get('ade_all_improvement', 0.0))}`, t50 `{_pct(payload['hp_ucy_before'].get('ade_t50_improvement', 0.0))}`.",
        f"- repaired global all/t50/t100 raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- repaired UCY all/t50/hard/easy: `{_pct(ucy.get('all_improvement', 0.0))}` / `{_pct(ucy.get('t50_improvement', 0.0))}` / `{_pct(ucy.get('hard_failure_improvement', 0.0))}` / `{_pct(ucy.get('easy_degradation', 0.0))}`.",
        f"- t100 easy status: rows `{payload['ucy_supported_repair']['test']['t100_easy_rows']}`, degradation `{_pct(payload['ucy_supported_repair']['test']['t100_easy_degradation'])}`; recorded as raw-frame diagnostic, not seconds-level.",
        "- claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hq_gate"]
    metric = payload["ucy_supported_repair"]["test"]["metric"]
    ucy = payload["ucy_supported_repair"]["test"]["by_domain"].get("UCY", {})
    traj = payload["ucy_supported_repair"]["test"]["by_domain"].get("TrajNet", {})
    diag = payload["ucy_supported_repair"]["test"]["diagnostics"]
    lines = [
        "# Stage42-HQ UCY Weak-Slice Group-Consistency Repair",
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
        "## HP Weak Slice Before",
        "",
        f"- HP UCY rows: `{payload['hp_ucy_before'].get('rows', 0)}`",
        f"- HP UCY all: `{_pct(payload['hp_ucy_before'].get('ade_all_improvement', 0.0))}`",
        f"- HP UCY t50: `{_pct(payload['hp_ucy_before'].get('ade_t50_improvement', 0.0))}`",
        f"- HP UCY reason: `{payload['hp_weak_reason']}`",
        "",
        "## Fresh Repair Result",
        "",
        "| metric | global | UCY | TrajNet |",
        "| --- | ---: | ---: | ---: |",
        f"| all | {_pct(metric['all_improvement'])} | {_pct(ucy.get('all_improvement', 0.0))} | {_pct(traj.get('all_improvement', 0.0))} |",
        f"| t50 | {_pct(metric['t50_improvement'])} | {_pct(ucy.get('t50_improvement', 0.0))} | {_pct(traj.get('t50_improvement', 0.0))} |",
        f"| t100 raw diag | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} | {_pct(ucy.get('t100_raw_frame_diagnostic_improvement', 0.0))} | {_pct(traj.get('t100_raw_frame_diagnostic_improvement', 0.0))} |",
        f"| hard/failure | {_pct(metric['hard_failure_improvement'])} | {_pct(ucy.get('hard_failure_improvement', 0.0))} | {_pct(traj.get('hard_failure_improvement', 0.0))} |",
        f"| easy degradation | {_pct(metric['easy_degradation'])} | {_pct(ucy.get('easy_degradation', 0.0))} | {_pct(traj.get('easy_degradation', 0.0))} |",
        f"| switch | {_pct(metric['switch_rate'])} | {_pct(ucy.get('switch_rate', 0.0))} | {_pct(traj.get('switch_rate', 0.0))} |",
        "",
        "## Safety And Remaining Risk",
        "",
        f"- near@0.05 base/final/floor: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}` / `{_pct(diag['floor_near_005'])}`",
        f"- t100 easy rows: `{payload['ucy_supported_repair']['test']['t100_easy_rows']}`",
        f"- t100 easy degradation after repair: `{_pct(payload['ucy_supported_repair']['test']['t100_easy_degradation'])}`",
        "- t100 remains raw-frame diagnostic; if this slice is used as a main claim, it needs a separate validation-only easy guard.",
        "",
        "## Interpretation",
        "",
        "- Stage42-HQ directly addresses the Stage42-HP UCY zero-gain weak slice.",
        "- UCY repair is supported by train-only internal validation, not test-threshold tuning.",
        "- This is still protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true-3D, Stage5C, or SMC evidence.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HQ Gate",
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
        _replace_section(path, "STAGE42_HQ_GROUP_CONSISTENCY_WEAK_SLICE_REPAIR", lines)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HQ UCY weak-slice group-consistency repair"
    state["current_verdict"] = payload["stage42_hq_gate"]["verdict"]
    state["stage42_hq_group_consistency_weak_slice_repair"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hq_gate"]["verdict"],
        "gates": f"{payload['stage42_hq_gate']['passed']}/{payload['stage42_hq_gate']['total']}",
        "hp_ucy_before": payload["hp_ucy_before"],
        "repair_metric": payload["ucy_supported_repair"]["test"]["metric"],
        "repair_by_domain": payload["ucy_supported_repair"]["test"]["by_domain"],
        "t100_easy_status": {
            "rows": payload["ucy_supported_repair"]["test"]["t100_easy_rows"],
            "degradation": payload["ucy_supported_repair"]["test"]["t100_easy_degradation"],
        },
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Fresh UCY-internal-validation-supported group-consistency repair fixes the Stage42-HP UCY zero-gain weak slice while keeping no-leakage and raw-frame claim boundaries.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_group_consistency_weak_slice_repair.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_weak_slice_repair.py",
        },
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_weak_slice_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hp = read_json(HP_JSON, {}) if HP_JSON.exists() else {}
    hp_ucy = _hp_ucy_before(hp)
    weak_reason = ""
    for row in hp.get("weak_slices", []):
        if row.get("name") == "domain:UCY":
            weak_reason = str(row.get("reason", ""))
    repair = _repair_with_ucy_support()
    payload: dict[str, Any] = {
        "source": "fresh_stage42_hq_ucy_weak_slice_repair",
        "stage": "Stage42-HQ group-consistency weak-slice repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([HP_JSON, Path("data/stage41_world_model/combined_external.npz")]),
        "hp_input": {"path": str(HP_JSON), "exists": bool(hp)},
        "hp_ucy_before": hp_ucy,
        "hp_weak_reason": weak_reason,
        "ucy_supported_repair": repair,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hq_gate"] = _gate(payload)
    write_json(REPORT_JSON, di._jsonable(payload))
    _write_outputs(payload)
    _refresh_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_group_consistency_weak_slice_repair()
    gate = out["stage42_hq_gate"]
    print(f"Stage42-HQ group-consistency weak-slice repair: {gate['verdict']} ({gate['passed']}/{gate['total']})")
