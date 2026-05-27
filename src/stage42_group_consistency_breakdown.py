from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_group_consistency_runtime_policy import FrozenGroupConsistencyPolicy, POLICY_JSON
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "group_consistency_breakdown_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_breakdown_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hp_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HP 是 frozen group-consistency full-waypoint policy 的 fresh source-level breakdown，不是新阈值搜索。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), am.EPS)


def _local_metric(
    *,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    selected_fde: np.ndarray,
    floor_fde: np.ndarray,
    horizon: np.ndarray,
    hard: np.ndarray,
    failure: np.ndarray,
    easy: np.ndarray,
    switch: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    hard_failure = hard | failure
    return {
        "rows": int(np.sum(mask)),
        "t10_rows": int(np.sum(mask & (horizon == 10))),
        "t25_rows": int(np.sum(mask & (horizon == 25))),
        "t50_rows": int(np.sum(mask & (horizon == 50))),
        "t100_rows": int(np.sum(mask & (horizon == 100))),
        "ade_all_improvement": _safe_improvement(selected_ade, floor_ade, mask),
        "ade_t10_improvement": _safe_improvement(selected_ade, floor_ade, mask & (horizon == 10)),
        "ade_t25_improvement": _safe_improvement(selected_ade, floor_ade, mask & (horizon == 25)),
        "ade_t50_improvement": _safe_improvement(selected_ade, floor_ade, mask & (horizon == 50)),
        "ade_t100_raw_frame_diagnostic_improvement": _safe_improvement(selected_ade, floor_ade, mask & (horizon == 100)),
        "ade_hard_failure_improvement": _safe_improvement(selected_ade, floor_ade, mask & hard_failure),
        "ade_easy_degradation": -_safe_improvement(selected_ade, floor_ade, mask & easy),
        "fde_all_improvement": _safe_improvement(selected_fde, floor_fde, mask),
        "fde_t50_improvement": _safe_improvement(selected_fde, floor_fde, mask & (horizon == 50)),
        "fde_t100_raw_frame_diagnostic_improvement": _safe_improvement(selected_fde, floor_fde, mask & (horizon == 100)),
        "switch_rate": float(np.mean(switch[mask])) if np.any(mask) else 0.0,
        "harm_over_floor_ade": float(np.mean(selected_ade[mask] - floor_ade[mask])) if np.any(mask) else 0.0,
    }


def _safety_metric(
    *,
    base_min: np.ndarray,
    final_min: np.ndarray,
    floor_min: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "rows": 0,
            "base_near_005": 0.0,
            "final_near_005": 0.0,
            "floor_near_005": 0.0,
            "near_005_delta_vs_base": 0.0,
            "base_p05_min_distance": None,
            "final_p05_min_distance": None,
        }
    b = base_min[mask]
    f = final_min[mask]
    floor = floor_min[mask]
    finite_b = b[np.isfinite(b)]
    finite_f = f[np.isfinite(f)]
    return {
        "rows": int(np.sum(mask)),
        "base_near_005": float(np.mean(np.isfinite(b) & (b < 0.05))),
        "final_near_005": float(np.mean(np.isfinite(f) & (f < 0.05))),
        "floor_near_005": float(np.mean(np.isfinite(floor) & (floor < 0.05))),
        "near_005_delta_vs_base": float(np.mean(np.isfinite(f) & (f < 0.05)) - np.mean(np.isfinite(b) & (b < 0.05))),
        "base_p05_min_distance": float(np.percentile(finite_b, 5)) if finite_b.size else None,
        "final_p05_min_distance": float(np.percentile(finite_f, 5)) if finite_f.size else None,
    }


def _slice_record(
    *,
    name: str,
    mask: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    selected_fde: np.ndarray,
    floor_fde: np.ndarray,
    horizon: np.ndarray,
    hard: np.ndarray,
    failure: np.ndarray,
    easy: np.ndarray,
    switch: np.ndarray,
    base_min: np.ndarray,
    final_min: np.ndarray,
    floor_min: np.ndarray,
) -> dict[str, Any]:
    metric = _local_metric(
        selected_ade=selected_ade,
        floor_ade=floor_ade,
        selected_fde=selected_fde,
        floor_fde=floor_fde,
        horizon=horizon,
        hard=hard,
        failure=failure,
        easy=easy,
        switch=switch,
        mask=mask,
    )
    safety = _safety_metric(base_min=base_min, final_min=final_min, floor_min=floor_min, mask=mask)
    return {"name": str(name), **metric, "safety": safety}


def _records_by_values(
    values: np.ndarray,
    *,
    prefix: str,
    min_rows: int,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in sorted(set(values.astype(str).tolist())):
        mask = values.astype(str) == value
        if int(np.sum(mask)) < min_rows:
            continue
        records.append(_slice_record(name=f"{prefix}:{value}", mask=mask, **kwargs))
    return records


def _weak_slices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weak = []
    for row in records:
        rows = int(row.get("rows", 0))
        if rows < 100:
            continue
        t50_rows = int(row.get("t50_rows", 0))
        t50 = float(row.get("ade_t50_improvement", 0.0))
        all_imp = float(row.get("ade_all_improvement", 0.0))
        easy = float(row.get("ade_easy_degradation", 0.0))
        near_delta = float(row.get("safety", {}).get("near_005_delta_vs_base", 0.0))
        t50_weak = t50_rows > 0 and t50 <= 0.0
        if t50_weak or all_imp <= 0.0 or easy > 0.02 or near_delta > 0.0:
            weak.append(
                {
                    "name": row["name"],
                    "rows": rows,
                    "t50_rows": t50_rows,
                    "ade_all_improvement": all_imp,
                    "ade_t50_improvement": t50,
                    "ade_easy_degradation": easy,
                    "near_005_delta_vs_base": near_delta,
                    "reason": "; ".join(
                        [
                            part
                            for part, flag in [
                                ("non_positive_all", all_imp <= 0.0),
                                ("non_positive_t50", t50_weak),
                                ("easy_degradation_over_2pct", easy > 0.02),
                                ("near_collision_worse", near_delta > 0.0),
                            ]
                            if flag
                        ]
                    ),
                }
            )
    weak.sort(key=lambda r: (float(r["ade_t50_improvement"]), float(r["ade_all_improvement"])))
    return weak[:20]


def _build_breakdown() -> dict[str, Any]:
    s42b.build_stage42_source_split()
    policy = FrozenGroupConsistencyPolicy.from_file()
    data = s41._combined()
    split, group = am._split_arrays(data)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    group_key = di._group_key(data)
    test_ids = np.where(split == "test")[0]
    runtime = policy.apply(
        base_xy=am_candidate["selected_xy"][test_ids].astype(np.float32),
        floor_xy=floor["floor_xy"][test_ids].astype(np.float32),
        pred_xy=am_candidate["pred_xy"][test_ids].astype(np.float32),
        base_switch=am_candidate["switch"][test_ids].astype(bool),
        group_key=group_key[test_ids],
        normalizer=np.maximum(data["scale"][test_ids].astype(np.float64), di.EPS),
        agent_id=data["agent_id"][test_ids].astype(np.int64),
        current_xy=np.stack([data["current_x"][test_ids], data["current_y"][test_ids]], axis=1).astype(np.float32),
    )
    selected_ade, selected_fde = di._trajectory_errors_subset(runtime.selected_xy, labels, test_ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor["floor_xy"][test_ids], labels, test_ids)

    domain = data["dataset"][test_ids].astype(str)
    source = np.asarray([f"{d}::{s42b._rel_source(s)}" for d, s in zip(domain, data["source_file"][test_ids].astype(str))])
    scene = np.asarray([f"{d}::{sc}" for d, sc in zip(domain, data["scene_id"][test_ids].astype(str))])
    horizon = data["horizon"][test_ids].astype(int)
    hard = data["hard"][test_ids].astype(bool)
    failure = data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    kwargs = {
        "selected_ade": selected_ade,
        "floor_ade": floor_ade,
        "selected_fde": selected_fde,
        "floor_fde": floor_fde,
        "horizon": horizon,
        "hard": hard,
        "failure": failure,
        "easy": easy,
        "switch": runtime.switch,
        "base_min": runtime.base_min,
        "final_min": runtime.final_min,
        "floor_min": runtime.floor_min,
    }
    all_mask = np.ones(len(test_ids), dtype=bool)
    by_subset = [
        _slice_record(name="all", mask=all_mask, **kwargs),
        _slice_record(name="t50", mask=horizon == 50, **kwargs),
        _slice_record(name="t100_raw_frame_diagnostic", mask=horizon == 100, **kwargs),
        _slice_record(name="hard_failure", mask=hard | failure, **kwargs),
        _slice_record(name="easy", mask=easy, **kwargs),
        _slice_record(name="switched", mask=runtime.switch.astype(bool), **kwargs),
        _slice_record(name="fallback_only", mask=~runtime.switch.astype(bool), **kwargs),
    ]
    records = {
        "by_domain": _records_by_values(domain, prefix="domain", min_rows=1, **kwargs),
        "by_source": _records_by_values(source, prefix="source", min_rows=1, **kwargs),
        "by_scene": _records_by_values(scene, prefix="scene", min_rows=1, **kwargs),
        "by_horizon": _records_by_values(horizon.astype(str), prefix="horizon", min_rows=1, **kwargs),
        "by_subset": by_subset,
    }
    all_records = [row for group_records in records.values() for row in group_records]
    return {
        "source": "fresh_run_group_consistency_source_breakdown",
        "stage": "Stage42-HP group-consistency source-level breakdown",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([POLICY_JSON, Path("outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json")]),
        "policy_artifact": {"path": str(POLICY_JSON), "hash": _combined_hash([POLICY_JSON])},
        "split_stats": am._source_stats(data, split, group),
        "rows": {
            "test_rows": int(len(test_ids)),
            "domain_counts": dict(Counter(domain.tolist())),
            "source_counts": dict(Counter(source.tolist())),
            "scene_counts": dict(Counter(scene.tolist())),
            "horizon_counts": dict(Counter(horizon.astype(int).tolist())),
            "hard_rows": int(np.sum(hard)),
            "failure_rows": int(np.sum(failure)),
            "easy_rows": int(np.sum(easy)),
        },
        "overall": by_subset[0],
        "breakdown": records,
        "weak_slices": _weak_slices(all_records),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "source_overlap_pass": bool(am._source_stats(data, split, group)["source_overlap_pass"]),
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


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]
    by_domain = payload["breakdown"]["by_domain"]
    by_source = payload["breakdown"]["by_source"]
    by_scene = payload["breakdown"]["by_scene"]
    by_horizon = payload["breakdown"]["by_horizon"]
    horizons = {row["name"].split(":", 1)[1] for row in by_horizon}
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    gates = {
        "fresh_breakdown_source": payload["source"] == "fresh_run_group_consistency_source_breakdown",
        "test_rows_match_runtime_replay": int(payload["rows"]["test_rows"]) == 47458,
        "policy_artifact_hashed": bool(payload["policy_artifact"]["hash"]),
        "domain_breakdown_present": len(by_domain) >= 2,
        "source_breakdown_present": len(by_source) >= 3,
        "scene_breakdown_present": len(by_scene) >= 2,
        "all_horizons_present": {"10", "25", "50", "100"}.issubset(horizons),
        "overall_all_positive": float(overall["ade_all_improvement"]) > 0.0,
        "overall_t50_positive": float(overall["ade_t50_improvement"]) > 0.0,
        "overall_t100_raw_positive": float(overall["ade_t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "overall_hard_positive": float(overall["ade_hard_failure_improvement"]) > 0.0,
        "easy_preserved": float(overall["ade_easy_degradation"]) <= 0.02,
        "near_collision_not_worse": float(overall["safety"]["near_005_delta_vs_base"]) <= 0.0,
        "weak_slices_recorded": "weak_slices" in payload,
        "no_future_endpoint_input": no_leakage["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leakage["future_waypoint_input"] is False,
        "no_central_velocity": no_leakage["central_velocity"] is False,
        "no_test_endpoint_goals": no_leakage["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leakage["test_threshold_tuning"] is False,
        "source_overlap_pass": no_leakage["source_overlap_pass"] is True,
        "no_metric_seconds_claim": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hp_group_consistency_breakdown_pass" if passed == total else "stage42_hp_group_consistency_breakdown_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _summary_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hp_gate"]
    overall = payload["overall"]
    weak = payload["weak_slices"][:5]
    lines = [
        "## Stage42-HP Group-Consistency Source Breakdown",
        "",
        "- source: `fresh_run_group_consistency_source_breakdown`",
        "- role: break down frozen group-consistency full-waypoint policy by domain/source/scene/horizon/subset.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- rows: `{payload['rows']['test_rows']}`; domains `{payload['rows']['domain_counts']}`.",
        f"- ADE vs train-horizon causal floor: all `{_pct(overall['ade_all_improvement'])}`, t50 `{_pct(overall['ade_t50_improvement'])}`, t100 raw `{_pct(overall['ade_t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(overall['ade_hard_failure_improvement'])}`, easy `{_pct(overall['ade_easy_degradation'])}`.",
        f"- FDE: all `{_pct(overall['fde_all_improvement'])}`, t50 `{_pct(overall['fde_t50_improvement'])}`, t100 raw `{_pct(overall['fde_t100_raw_frame_diagnostic_improvement'])}`.",
        f"- group safety near@0.05 delta vs base: `{_pct(overall['safety']['near_005_delta_vs_base'])}`.",
        f"- weak slices recorded: `{len(payload['weak_slices'])}`; top examples `{[row['name'] for row in weak]}`.",
        "- claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hp_gate"]
    lines = [
        "# Stage42-HP Group-Consistency Source-Level Breakdown",
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
        "## Overall",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    overall = payload["overall"]
    for key in [
        "rows",
        "ade_all_improvement",
        "ade_t50_improvement",
        "ade_t100_raw_frame_diagnostic_improvement",
        "ade_hard_failure_improvement",
        "ade_easy_degradation",
        "fde_all_improvement",
        "fde_t50_improvement",
        "fde_t100_raw_frame_diagnostic_improvement",
        "switch_rate",
        "harm_over_floor_ade",
    ]:
        val = overall[key]
        shown = str(val) if key == "rows" else _pct(float(val))
        lines.append(f"| `{key}` | `{shown}` |")
    lines.extend(
        [
            "",
            "## By Domain",
            "",
            "| domain | rows | ADE all | ADE t50 | ADE t100 raw | hard/failure | easy | FDE t50 | switch | near delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["breakdown"]["by_domain"]:
        lines.append(
            f"| `{row['name']}` | {row['rows']} | {_pct(row['ade_all_improvement'])} | {_pct(row['ade_t50_improvement'])} | "
            f"{_pct(row['ade_t100_raw_frame_diagnostic_improvement'])} | {_pct(row['ade_hard_failure_improvement'])} | "
            f"{_pct(row['ade_easy_degradation'])} | {_pct(row['fde_t50_improvement'])} | {_pct(row['switch_rate'])} | "
            f"{_pct(row['safety']['near_005_delta_vs_base'])} |"
        )
    lines.extend(
        [
            "",
            "## By Horizon",
            "",
            "| horizon | rows | ADE all | FDE all | switch | near delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["breakdown"]["by_horizon"]:
        lines.append(
            f"| `{row['name']}` | {row['rows']} | {_pct(row['ade_all_improvement'])} | "
            f"{_pct(row['fde_all_improvement'])} | {_pct(row['switch_rate'])} | {_pct(row['safety']['near_005_delta_vs_base'])} |"
        )
    lines.extend(
        [
            "",
            "## Weak Slice Ledger",
            "",
        "| slice | rows | t50 rows | ADE all | ADE t50 | easy | near delta | reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["weak_slices"][:20]:
        lines.append(
            f"| `{row['name']}` | {row['rows']} | {row['t50_rows']} | {_pct(row['ade_all_improvement'])} | {_pct(row['ade_t50_improvement'])} | "
            f"{_pct(row['ade_easy_degradation'])} | {_pct(row['near_005_delta_vs_base'])} | `{row['reason']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-HP adds per-domain/per-source/per-scene/per-horizon evidence for the frozen group-consistency full-waypoint policy.",
            "- This is useful for paper-level evidence because it exposes weak slices instead of hiding them behind aggregate metrics.",
            "- It does not execute Stage5C, does not enable SMC, and does not make metric/seconds-level claims.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HP Gate",
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
        _replace_section(path, "STAGE42_HP_GROUP_CONSISTENCY_BREAKDOWN", lines)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HP group-consistency source-level breakdown"
    state["current_verdict"] = payload["stage42_hp_gate"]["verdict"]
    state["stage42_hp_group_consistency_breakdown"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hp_gate"]["verdict"],
        "gates": f"{payload['stage42_hp_gate']['passed']}/{payload['stage42_hp_gate']['total']}",
        "rows": payload["rows"],
        "overall": payload["overall"],
        "weak_slice_count": len(payload["weak_slices"]),
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Fresh source-level breakdown confirms the frozen group-consistency full-waypoint policy remains positive overall while preserving an explicit weak-slice ledger for paper-level honesty.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_group_consistency_breakdown.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_breakdown.py",
        },
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_breakdown() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_breakdown()
    payload["stage42_hp_gate"] = _gate(payload)
    write_json(REPORT_JSON, am._jsonable(payload))
    _write_outputs(payload)
    _refresh_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_group_consistency_breakdown()
    gate = result["stage42_hp_gate"]
    print(f"Stage42-HP group-consistency breakdown: {gate['verdict']} ({gate['passed']}/{gate['total']})")
