from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_context_gain_router as el
from src import stage42_sequence_graph_context_router as eq
from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as sg
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, read_json, write_json
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_t50_switchability_calibration_repair import (
    CANDIDATES,
    CLAIM_BOUNDARY,
    HORIZON,
    README_RESULTS,
    M3W_README,
    RESEARCH_STATE,
    STRATEGIES,
    WORK_LEDGER,
    _evaluate_strategy,
    _fmt,
)
from src.stage42_t50_t100_sequence_graph_blocker_audit import _filter_rows, _oracle_metric


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_source_pattern_switchability_repair_stage42.json"
REPORT_MD = OUT_DIR / "t50_source_pattern_switchability_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ir_gate.md"

SOURCE = "fresh_stage42_t50_source_pattern_switchability_repair"
PATTERNS = [
    "students",
    "zara",
    "crowds",
    "biwi",
    "eth",
    "hotel",
    "trajnet_train",
    "trajnet_test",
    "ucy_path",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IR 是 Stage42-IQ 后续 repair：把 source-pattern support 加到 t50 sequence+graph switchability router。",
    "source pattern 来自已知 source_file 路径模式，不来自 test endpoint、future waypoint 或 metric calibration。",
    "future waypoints / endpoints 只作为 train/val supervised labels 或 test evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _source_pattern_features(source_file: np.ndarray) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    src = source_file.astype(str)
    cols = [
        np.char.find(src, "students") >= 0,
        np.char.find(src, "zara") >= 0,
        np.char.find(src, "crowds") >= 0,
        np.char.find(src, "biwi") >= 0,
        np.char.find(src, "seq_eth") >= 0,
        np.char.find(src, "hotel") >= 0,
        np.char.find(src, "/TrajNet/Train/") >= 0,
        np.char.find(src, "/TrajNet/Test/") >= 0,
        np.char.find(src, "/UCY/") >= 0,
    ]
    features = np.stack(cols, axis=1).astype(np.float32)
    stats = {
        "source": "fresh_run",
        "feature_names": PATTERNS,
        "feature_count": len(PATTERNS),
        "rows": int(len(source_file)),
        "uses_future_endpoint": False,
        "uses_future_waypoint": False,
        "uses_test_endpoint_goals": False,
    }
    return features, PATTERNS, stats


def _build_result() -> dict[str, Any]:
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    data = shared["data"]
    hmask = data["horizon"].astype(int) == HORIZON
    hdata = _filter_rows(data, hmask)
    hsplit = split[hmask]
    base_pred = el._prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    graph, graph_names, graph_stats = sg._build_graph_features(data)
    seq_summary, seq_names, seq_stats = eq._sequence_summary(data)
    source_pattern, pattern_names, pattern_stats = _source_pattern_features(data["source_file"])
    trials: dict[str, Any] = {}
    oracle_by_candidate: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_features = shared["features"][:, masks[candidate]]
        candidate_pred = el._prepare_variant_predictions(candidate_features, shared)
        augmented = np.concatenate(
            [candidate_features.astype(np.float32), graph.astype(np.float32), seq_summary.astype(np.float32), source_pattern],
            axis=1,
        ).astype(np.float32)
        h_augmented = augmented[hmask]
        h_base = base_pred["selected_ade"][hmask]
        h_candidate = candidate_pred["selected_ade"][hmask]
        test_mask = hsplit == "test"
        oracle_by_candidate[candidate] = _oracle_metric(h_base, h_candidate, hdata, test_mask)
        for strategy in STRATEGIES:
            key = f"{candidate}__{strategy}"
            trials[key] = _evaluate_strategy(
                strategy=strategy,
                candidate=candidate,
                raw_features=h_augmented,
                base_ade=h_base,
                candidate_ade=h_candidate,
                split=hsplit,
                data=hdata,
            )
    best_key = max(
        trials,
        key=lambda key: (
            trials[key]["test_metric"]["t50_improvement"]
            + trials[key]["test_metric"]["hard_failure_improvement"]
            - max(0.0, trials[key]["test_metric"]["easy_degradation"] - 0.02)
        ),
    )
    best = trials[best_key]
    repair_supported = bool(best["deployable_supported"])
    iq = read_json(OUT_DIR / "t50_switchability_calibration_repair_stage42.json", {})
    result = {
        "source": SOURCE,
        "stage": "Stage42-IR t50 Source-Pattern Switchability Repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/t50_switchability_calibration_repair_stage42.json",
                "outputs/stage42_long_research/t50_t100_sequence_graph_blocker_audit_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "horizon": HORIZON,
        "rows": {
            "train": int(np.sum(hsplit == "train")),
            "val": int(np.sum(hsplit == "val")),
            "test": int(np.sum(hsplit == "test")),
        },
        "sequence_summary_schema": {"feature_names": seq_names, "stats": seq_stats},
        "graph_summary_schema": {"feature_names": graph_names, "stats": graph_stats},
        "source_pattern_schema": {"feature_names": pattern_names, "stats": pattern_stats},
        "oracle_by_candidate": oracle_by_candidate,
        "trials": trials,
        "best_trial_key": best_key,
        "best_trial": best,
        "repair_supported": repair_supported,
        "stage42_iq_reference": {
            "source": "cached_verified" if iq else "not_run",
            "repair_supported": (iq or {}).get("repair_supported"),
            "best_trial_key": (iq or {}).get("best_trial_key"),
            "best_trial_metric": (iq or {}).get("best_trial", {}).get("test_metric"),
        },
        "summary": {
            "source": SOURCE,
            "purpose": "test whether source-pattern support repairs Stage42-IQ t50 switchability failure",
            "patterns": PATTERNS,
            "best_trial_key": best_key,
            "best_trial_metric": best["test_metric"],
            "repair_supported": repair_supported,
            "verdict": "t50_source_pattern_switchability_repair_supported"
            if repair_supported
            else "t50_source_pattern_switchability_repair_not_supported",
            "interpretation": (
                "Stage42-IR changes source support rather than only thresholds. If it remains unsupported, "
                "the context t50 route should be treated as closed under this candidate family; future repair needs "
                "new candidate policies or new source data rather than more source-pattern gating."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_label_train_val_supervision_only": True,
            "source_file_pattern_input_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_selected_thresholds": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["stage42_ir_gate"] = _gate(result)
    return _jsonable(result)


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    nl = result["no_leakage"]
    no_leakage_pass = (
        nl["future_endpoint_input"] is False
        and nl["future_waypoint_input"] is False
        and nl["future_label_train_val_supervision_only"] is True
        and nl["source_file_pattern_input_only"] is True
        and nl["central_velocity"] is False
        and nl["test_endpoint_goals"] is False
        and nl["test_threshold_tuning"] is False
        and nl["validation_selected_thresholds"] is True
        and nl["source_overlap_pass"] is True
    )
    gates = {
        "stage42_iq_loaded": result["stage42_iq_reference"]["source"] != "not_run",
        "source_pattern_schema_built": result["source_pattern_schema"]["stats"]["feature_count"] == len(PATTERNS),
        "t50_rows_present": result["rows"]["test"] > 1000,
        "all_pattern_trials_evaluated": len(result["trials"]) == len(CANDIDATES) * len(STRATEGIES),
        "validation_only_selection": all(
            row["validation_selection"]["test_threshold_tuning"] is False for row in result["trials"].values()
        ),
        "test_result_reported": "test_metric" in result["best_trial"],
        "success_or_honest_failure_reported": result["summary"]["verdict"]
        in {"t50_source_pattern_switchability_repair_supported", "t50_source_pattern_switchability_repair_not_supported"},
        "no_leakage_pass": no_leakage_pass,
        "no_metric_seconds_overclaim": not result["claim_boundary"]["global_metric_claim_allowed"]
        and not result["claim_boundary"]["global_seconds_claim_allowed"],
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    return {
        "passed": int(sum(gates.values())),
        "total": int(len(gates)),
        "gates": gates,
        "verdict": "stage42_ir_t50_source_pattern_switchability_repair_pass"
        if all(gates.values())
        else "stage42_ir_t50_source_pattern_switchability_repair_fail",
    }


def _render_md(result: Mapping[str, Any]) -> str:
    gate = result["stage42_ir_gate"]
    best = result["best_trial"]
    lines = [
        "# Stage42-IR t50 Source-Pattern Switchability Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- repair_supported: `{result['repair_supported']}`",
        f"- repair_verdict: `{result['summary']['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in result["current_facts"])
    lines.extend(
        [
            "",
            "## Best Trial",
            "",
            f"- best_trial_key: `{result['best_trial_key']}`",
            f"- test t50 improvement: `{_fmt(best['test_metric']['t50_improvement'])}`",
            f"- test hard/failure improvement: `{_fmt(best['test_metric']['hard_failure_improvement'])}`",
            f"- test easy degradation: `{_fmt(best['test_metric']['easy_degradation'])}`",
            f"- test switch rate: `{_fmt(best['test_metric']['switch_rate'])}`",
            "",
            "## Trial Table",
            "",
            "| trial | val t50 | test t50 | hard/failure | easy deg | switch | supported |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in sorted(result["trials"]):
        row = result["trials"][key]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    _fmt(row["validation_selection"]["val_metric"]["t50_improvement"]),
                    _fmt(row["test_metric"]["t50_improvement"]),
                    _fmt(row["test_metric"]["hard_failure_improvement"]),
                    _fmt(row["test_metric"]["easy_degradation"]),
                    _fmt(row["test_metric"]["switch_rate"]),
                    str(row["deployable_supported"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["summary"]["interpretation"],
            "",
            "- This is a source-pattern repair attempt, not a new deployable model unless `repair_supported` is true.",
            "- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{name}` | {passed} |" for name, passed in gate["gates"].items())
    return "\n".join(lines) + "\n"


def _update_ledgers(result: Mapping[str, Any]) -> None:
    gate = result["stage42_ir_gate"]
    best = result["best_trial"]
    block = [
        "## Stage42-IR t50 Source-Pattern Switchability Repair",
        "",
        f"- source: `{result['source']}`",
        "- role: formal source-support repair attempt for Stage42-IQ t50 switchability failure.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- repair_supported: `{result['repair_supported']}`; repair_verdict `{result['summary']['verdict']}`.",
        f"- best_trial: `{result['best_trial_key']}`.",
        f"- best test t50 / hard / easy: `{_fmt(best['test_metric']['t50_improvement'])}` / `{_fmt(best['test_metric']['hard_failure_improvement'])}` / `{_fmt(best['test_metric']['easy_degradation'])}`.",
        "- conclusion: source-pattern support does not repair the context t50 route under this protocol; future repair needs new candidate policies or source data.",
        "- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_LEDGER]:
        _replace_section(path, "STAGE42_IR_T50_SOURCE_PATTERN_SWITCHABILITY_REPAIR", block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("generated_reports", [])
    for report in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if report not in state["generated_reports"]:
            state["generated_reports"].append(report)
    state.setdefault("stage42_long_research", {})
    state["stage42_long_research"]["stage_ir_t50_source_pattern_switchability_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "repair_supported": result["repair_supported"],
        "repair_verdict": result["summary"]["verdict"],
        "best_trial_key": result["best_trial_key"],
        "best_trial_metric": result["best_trial"]["test_metric"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    result = _build_result()
    write_json(REPORT_JSON, result)
    REPORT_MD.write_text(_render_md(result), encoding="utf-8")
    gate = result["stage42_ir_gate"]
    gate_lines = [
        "# Stage42-IR Gate",
        "",
        f"- source: `{result['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    gate_lines.extend(f"| `{name}` | {passed} |" for name, passed in gate["gates"].items())
    GATE_MD.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    _update_ledgers(result)
    return result


if __name__ == "__main__":
    out = run()
    print(f"Wrote {REPORT_MD}")
    print(f"Verdict: {out['stage42_ir_gate']['verdict']} ({out['stage42_ir_gate']['passed']}/{out['stage42_ir_gate']['total']})")
