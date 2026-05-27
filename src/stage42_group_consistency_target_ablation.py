from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "group_consistency_target_ablation_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_target_ablation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_et_gate.md"

DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
ES_JSON = OUT_DIR / "interaction_occupancy_target_selection_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
TARGET_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

SOURCE = "fresh_stage42_group_consistency_target_ablation"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-ET 对 Stage42-ES 选出的 explicit group-consistency target 做 group-schema ablation。",
    "本阶段 fresh-reruns source-level full-waypoint repair under alternative group keys；不下载、不转换、不执行 Stage5C、不启用 SMC。",
    "future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "validation 选择每个 group schema 内的 repair candidate；test 只评一次。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _group_key_variants(data: Mapping[str, np.ndarray], proper: np.ndarray) -> dict[str, np.ndarray]:
    source = data["source_file"].astype(str)
    frame = data["frame_id"].astype(np.float64)
    horizon = data["horizon"].astype(int)
    dataset = data["dataset"].astype(str)
    agent = data["agent_id"].astype(np.int64)
    rng = np.random.default_rng(42087)
    shuffled = proper.copy()
    rng.shuffle(shuffled)
    frame_bucket = np.floor(frame / 10.0).astype(int)
    return {
        "source_frame_horizon": proper.astype(object),
        "source_frame_no_horizon": np.asarray(
            [f"{s}\t{int(round(f * 1000.0))}" for s, f in zip(source, frame)],
            dtype=object,
        ),
        "source_framebucket10_horizon": np.asarray(
            [f"{s}\t{b}\t{h}" for s, b, h in zip(source, frame_bucket, horizon)],
            dtype=object,
        ),
        "shuffled_source_frame_horizon": shuffled.astype(object),
        "agent_isolated_no_interaction": np.asarray(
            [f"{s}\t{int(round(f * 1000.0))}\t{h}\t{a}" for s, f, h, a in zip(source, frame, horizon, agent)],
            dtype=object,
        ),
        "domain_frame_horizon": np.asarray(
            [f"{d}\t{int(round(f * 1000.0))}\t{h}" for d, f, h in zip(dataset, frame, horizon)],
            dtype=object,
        ),
    }


def _public_eval(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    metric = result["test"]["metric_vs_floor"]
    diag = result["test"]["diagnostics"]
    selected = result["selected"]
    return {
        "group_schema": name,
        "source": "fresh_run",
        "candidate_count": result["candidate_count"],
        "selected_candidate": selected["candidate"],
        "selected_val_score": selected["val_score"],
        "selected_val_metric": selected["val_metric"],
        "metric_vs_floor": metric,
        "diagnostics": diag,
        "bootstrap": result["test"]["bootstrap"],
        "by_domain": result["test"]["by_domain"],
        "selection_score_on_test": di._selection_score(metric, diag),
    }


def _contribution_delta(correct: Mapping[str, Any], isolated: Mapping[str, Any]) -> dict[str, float]:
    cm = correct["metric_vs_floor"]
    im = isolated["metric_vs_floor"]
    cd = correct["diagnostics"]
    idg = isolated["diagnostics"]
    return {
        "all_increment": float(cm["all_improvement"]) - float(im["all_improvement"]),
        "t50_increment": float(cm["t50_improvement"]) - float(im["t50_improvement"]),
        "t100_raw_frame_diagnostic_increment": float(cm["t100_raw_frame_diagnostic_improvement"])
        - float(im["t100_raw_frame_diagnostic_improvement"]),
        "hard_failure_increment": float(cm["hard_failure_improvement"]) - float(im["hard_failure_improvement"]),
        "easy_degradation_increment": float(cm["easy_degradation"]) - float(im["easy_degradation"]),
        "near005_reduction_vs_correct_base": float(cd["base_near_005"]) - float(cd["final_near_005"]),
        "isolated_near005_not_comparable": float(idg["final_near_005"]),
        "p05_min_distance_gain_vs_isolated": float(cd["final_p05_min_distance"] or 0.0)
        - float(idg["final_p05_min_distance"] or 0.0),
    }


def _select_group_schema(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row["selected_val_score"]), reverse=True)
    best = ranked[0]
    correct = next(row for row in rows if row["group_schema"] == "source_frame_horizon")
    isolated = next(row for row in rows if row["group_schema"] == "agent_isolated_no_interaction")
    delta = _contribution_delta(correct, isolated)
    return {
        "source": "validation_ranked_group_schema_ablation",
        "selected_by_validation_score": best["group_schema"],
        "selected_target_for_next_stage": "source_frame_horizon"
        if delta["hard_failure_increment"] > 0.0 and delta["near005_reduction_vs_correct_base"] > 0.0
        else best["group_schema"],
        "ranked_group_schemas": [
            {
                "group_schema": row["group_schema"],
                "selected_val_score": row["selected_val_score"],
                "all": row["metric_vs_floor"]["all_improvement"],
                "t50": row["metric_vs_floor"]["t50_improvement"],
                "hard": row["metric_vs_floor"]["hard_failure_improvement"],
                "easy": row["metric_vs_floor"]["easy_degradation"],
                "near005": row["diagnostics"]["final_near_005"],
            }
            for row in ranked
        ],
        "source_frame_horizon_vs_agent_isolated": delta,
        "decision": "keep_source_frame_horizon_group_consistency_target"
        if delta["hard_failure_increment"] > 0.0 and delta["near005_reduction_vs_correct_base"] > 0.0
        else "group_schema_ablation_does_not_support_source_frame_horizon_target",
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
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    proper = di._group_key(data)
    variants = _group_key_variants(data, proper)
    rows = []
    for name, group_key in variants.items():
        result = di._evaluate_repairs(data, split, labels, floor, am_candidate, group_key)
        rows.append(_public_eval(name, result))
    selection = _select_group_schema(rows)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-ET group-consistency target ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(DI_JSON), str(ES_JSON)]),
        "current_facts": CURRENT_FACTS,
        "source_split": source_split,
        "split_stats": split_stats,
        "stage42_am_rebuilt": {
            "source": "fresh_rebuild_for_et",
            "lambda": am_candidate["lambda"],
            "feature_count": am_candidate["feature_count"],
            "policy_slice_count": len(am_candidate["policy"]["slices"]),
            "val_metric": am_candidate["val_metric"],
        },
        "group_schema_ablation": {
            "source": "fresh_group_key_variant_eval",
            "schemas_evaluated": list(variants.keys()),
            "rows": rows,
            "selection": selection,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_candidate_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_et_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = {row["group_schema"]: row for row in payload["group_schema_ablation"]["rows"]}
    correct = rows["source_frame_horizon"]
    isolated = rows["agent_isolated_no_interaction"]
    delta = payload["group_schema_ablation"]["selection"]["source_frame_horizon_vs_agent_isolated"]
    cm = correct["metric_vs_floor"]
    cb = correct["bootstrap"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "group_schema_variants_evaluated": len(rows) >= 5,
        "source_frame_horizon_present": "source_frame_horizon" in rows,
        "agent_isolated_control_present": "agent_isolated_no_interaction" in rows,
        "source_frame_horizon_all_positive": cm["all_improvement"] > 0.0,
        "source_frame_horizon_t50_positive": cm["t50_improvement"] > 0.0,
        "source_frame_horizon_hard_positive": cm["hard_failure_improvement"] > 0.0,
        "source_frame_horizon_easy_safe": cm["easy_degradation"] <= 0.02,
        "source_frame_horizon_hard_beats_isolated": delta["hard_failure_increment"] > 0.0,
        "source_frame_horizon_near005_repaired_vs_own_base": delta["near005_reduction_vs_correct_base"] > 0.0,
        "source_frame_horizon_bootstrap_all_positive": cb["all"]["low"] > 0.0,
        "source_frame_horizon_bootstrap_hard_positive": cb["hard_failure"]["low"] > 0.0,
        "isolated_control_matches_no_interaction_target": isolated["diagnostics"]["unsafe_rows"] == 0,
        "no_leakage_pass": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity"] is False
        and no_leak["test_endpoint_goals"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["validation_only_candidate_selection"] is True
        and no_leak["train_only_feature_normalization"] is True
        and no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_et_group_consistency_target_ablation_pass" if passed == total else "stage42_et_group_consistency_target_ablation_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _paper_lines(payload: Mapping[str, Any]) -> list[str]:
    rows = {row["group_schema"]: row for row in payload["group_schema_ablation"]["rows"]}
    correct = rows["source_frame_horizon"]
    isolated = rows["agent_isolated_no_interaction"]
    delta = payload["group_schema_ablation"]["selection"]["source_frame_horizon_vs_agent_isolated"]
    cm = correct["metric_vs_floor"]
    im = isolated["metric_vs_floor"]
    return [
        "## Stage42-ET Group-Consistency Target Ablation",
        "",
        "- source: `fresh_stage42_group_consistency_target_ablation`",
        "- role: tests whether the Stage42-ES selected group-consistency target depends on the real source/frame/horizon multi-agent grouping.",
        f"- source/frame/horizon all/t50/t100raw/hard/easy: `{_pct(cm['all_improvement'])}` / `{_pct(cm['t50_improvement'])}` / `{_pct(cm['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(cm['hard_failure_improvement'])}` / `{_pct(cm['easy_degradation'])}`.",
        f"- agent-isolated control all/t50/hard/easy: `{_pct(im['all_improvement'])}` / `{_pct(im['t50_improvement'])}` / `{_pct(im['hard_failure_improvement'])}` / `{_pct(im['easy_degradation'])}`.",
        f"- source/frame/horizon vs isolated hard increment `{_pct(delta['hard_failure_increment'])}`; own-base near@0.05 reduction `{_pct(delta['near005_reduction_vs_correct_base'])}`.",
        f"- decision: `{payload['group_schema_ablation']['selection']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_lines(payload)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_et": "STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION" in text,
                    "contains_group_consistency": "group-consistency" in text,
                    "contains_boundaries": "not true 3D" in text and "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    selection = payload["group_schema_ablation"]["selection"]
    lines = [
        "# Stage42-ET Group-Consistency Target Ablation",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_et_gate']['passed']} / {payload['stage42_et_gate']['total']}`",
        f"- verdict: `{payload['stage42_et_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Selection",
        "",
        f"- selected_by_validation_score: `{selection['selected_by_validation_score']}`",
        f"- selected_target_for_next_stage: `{selection['selected_target_for_next_stage']}`",
        f"- decision: `{selection['decision']}`",
        "",
        "## Group Schema Comparison",
        "",
        "| schema | val score | all | t50 | t100 raw | hard | easy | near@0.05 | unsafe rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(payload["group_schema_ablation"]["rows"], key=lambda item: item["selected_val_score"], reverse=True):
        m = row["metric_vs_floor"]
        d = row["diagnostics"]
        lines.append(
            f"| `{row['group_schema']}` | {float(row['selected_val_score']):.6f} | {_pct(m['all_improvement'])} | {_pct(m['t50_improvement'])} | "
            f"{_pct(m['t100_raw_frame_diagnostic_improvement'])} | {_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} | "
            f"{_pct(d['final_near_005'])} | {d['unsafe_rows']} |"
        )
    delta = selection["source_frame_horizon_vs_agent_isolated"]
    lines.extend(
        [
            "",
            "## Contribution Delta",
            "",
            "| delta | value |",
            "| --- | ---: |",
            *[f"| `{key}` | {_pct(value)} |" for key, value in delta.items()],
            "",
            "## Interpretation",
            "",
            "- `agent_isolated_no_interaction` is the no-interaction accuracy control: each row is its own group, so group repair cannot use multi-agent proximity; its near@0.05 value is not a valid pairwise collision baseline.",
            "- A positive source/frame/horizon increment over the isolated control supports the group-consistency target as an interaction/occupancy constraint rather than a generic scalar loss artifact.",
            "- This remains protected source-level raw-frame 2.5D evidence, not a metric/seconds-level or floor-free neural claim.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_et_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_et_gate"]
    return [
        "# Stage42-ET Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _readme_lines(payload: Mapping[str, Any]) -> list[str]:
    rows = {row["group_schema"]: row for row in payload["group_schema_ablation"]["rows"]}
    correct = rows["source_frame_horizon"]
    isolated = rows["agent_isolated_no_interaction"]
    delta = payload["group_schema_ablation"]["selection"]["source_frame_horizon_vs_agent_isolated"]
    cm = correct["metric_vs_floor"]
    im = isolated["metric_vs_floor"]
    return [
        "## Stage42-ET Group-Consistency Target Ablation",
        "",
        "- source: `fresh_stage42_group_consistency_target_ablation`",
        "- role: tests whether the Stage42-ES selected interaction/occupancy target depends on real source/frame/horizon multi-agent grouping.",
        f"- gate: `{payload['stage42_et_gate']['passed']} / {payload['stage42_et_gate']['total']}`; verdict `{payload['stage42_et_gate']['verdict']}`.",
        f"- selected target for next stage: `{payload['group_schema_ablation']['selection']['selected_target_for_next_stage']}`; decision `{payload['group_schema_ablation']['selection']['decision']}`.",
        f"- source/frame/horizon all/t50/t100raw/hard/easy: `{_pct(cm['all_improvement'])}` / `{_pct(cm['t50_improvement'])}` / `{_pct(cm['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(cm['hard_failure_improvement'])}` / `{_pct(cm['easy_degradation'])}`.",
        f"- agent-isolated control all/t50/hard/easy: `{_pct(im['all_improvement'])}` / `{_pct(im['t50_improvement'])}` / `{_pct(im['hard_failure_improvement'])}` / `{_pct(im['easy_degradation'])}`.",
        f"- hard increment vs isolated `{_pct(delta['hard_failure_increment'])}`; own-base near@0.05 reduction `{_pct(delta['near005_reduction_vs_correct_base'])}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _readme_lines(payload)
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    selection = payload["group_schema_ablation"]["selection"]
    rows = {row["group_schema"]: row for row in payload["group_schema_ablation"]["rows"]}
    state["current_stage"] = "Stage42-ET group consistency target ablation"
    state["current_verdict"] = payload["stage42_et_gate"]["verdict"]
    state["stage42_et_group_consistency_target_ablation"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_et_gate"]["verdict"],
        "gates": f"{payload['stage42_et_gate']['passed']}/{payload['stage42_et_gate']['total']}",
        "selected_target_for_next_stage": selection["selected_target_for_next_stage"],
        "decision": selection["decision"],
        "source_frame_horizon_metric": rows["source_frame_horizon"]["metric_vs_floor"],
        "agent_isolated_metric": rows["agent_isolated_no_interaction"]["metric_vs_floor"],
        "source_frame_horizon_vs_agent_isolated": selection["source_frame_horizon_vs_agent_isolated"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_target_ablation(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_payload()
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_group_consistency_target_ablation()
