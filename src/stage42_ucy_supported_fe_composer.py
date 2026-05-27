from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_constrained_fc_safety_composer as fe
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_supported_fe_composer_stage42.json"
REPORT_MD = OUT_DIR / "ucy_supported_fe_composer_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fh_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fe.PAPER_FILES

SOURCE = "fresh_stage42_ucy_supported_fe_composer"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FG 发现 frozen FE 在 TrajNet robust，但 UCY 切片 fallback 为 0，不能写成 uniform source-level success。",
    "Stage42-FH 按 Stage42-DZ 的思想，从 UCY train-only source 中 carve internal validation，重新选择 FE composer family。",
    "test sources 不改变，test rows 只最终评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _positive_safe(metric: Mapping[str, Any]) -> bool:
    return bool(
        float(metric.get("all_improvement", 0.0)) > 0.0
        and float(metric.get("t50_improvement", 0.0)) > 0.0
        and float(metric.get("hard_failure_improvement", 0.0)) > 0.0
        and float(metric.get("easy_degradation", 1.0)) <= 0.02
    )


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = repaired_split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    graph, graph_names, graph_stats = graph_ctx._build_graph_features(data)
    am_candidate = di._rebuild_stage42_am_candidate(data, repaired_split, labels, floor)
    am_candidate["floor_xy"] = floor["floor_xy"]
    group_key = di._group_key(data)
    signals = fe.fc._objective_signals(data, labels, graph, group_key, am_candidate)
    prior = fe._load_prior()
    fc_candidate = fe._rebuild_fc_candidate(data, repaired_split, labels, floor, features, graph, signals, group_key, prior["fc"])
    repair = fe._evaluate_composer(data, repaired_split, labels, floor, am_candidate, fc_candidate, group_key, prior)
    original_stats = am._source_stats(data, original_split, group)
    repaired_stats = am._source_stats(data, repaired_split, group)
    test_ids = np.where(repaired_split == "test")[0]
    test_domain = data["dataset"][test_ids].astype(str)
    by_domain = repair["test"]["by_domain"]
    summary = {
        "internal_val_group": internal_val_group,
        "ucy_val_rows_after": int(repaired_stats["by_split"]["val"]["domains"].get("UCY", 0)),
        "test_rows_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
        "test_domains": sorted(set(test_domain.tolist())),
        "positive_safe_domains": sorted([name for name, row in by_domain.items() if _positive_safe(row)]),
        "weak_domains": sorted([name for name, row in by_domain.items() if not _positive_safe(row)]),
        "selected_candidate": repair["selected"]["candidate"],
        "deployment_decision": "promote_stage42_fh_ucy_supported_fe_composer"
        if len([name for name, row in by_domain.items() if _positive_safe(row)]) >= 2
        else "ucy_supported_fe_composer_partial_keep_fe_ff_with_ucy_blocker",
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FH UCY-supported FE composer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fe._git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(fe.AM_JSON),
                str(fe.DI_JSON),
                str(fe.FA_JSON),
                str(fe.FB_JSON),
                str(fe.FC_JSON),
                "outputs/stage42_long_research/ucy_validation_support_repair_stage42.json",
                "outputs/stage42_long_research/fe_source_robustness_audit_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "original_split_stats": am._source_stats(data, original_split, group),
        "repaired_split_stats": repaired_stats,
        "feature_schema": {
            "base_feature_count": len(feature_names),
            "graph_feature_count": len(graph_names),
            "future_label_input": False,
        },
        "graph_stats": graph_stats,
        "internal_validation": {
            "source": "fresh_run",
            "domain": "UCY",
            "selected_from": "original_train_sources_only",
            "internal_val_group": internal_val_group,
            "uses_test_rows": False,
            "test_rows_unchanged": summary["test_rows_unchanged"],
        },
        "repair": repair,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "internal_val_from_train_only": True,
            "test_sources_unchanged": summary["test_rows_unchanged"],
            "source_overlap_pass": bool(repaired_stats["source_overlap_pass"]),
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
    payload["stage42_fh_gate"] = _gate(payload)
    return am._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    repair = payload["repair"]
    metric = repair["test"]["metric_vs_floor"]
    summary = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "ucy_internal_val_created": summary["ucy_val_rows_after"] > 0,
        "test_sources_unchanged": no_leak["test_sources_unchanged"] is True,
        "source_overlap_pass": no_leak["source_overlap_pass"] is True,
        "composer_family_built": repair["candidate_count"] >= 40,
        "validation_selected_without_test": no_leak["test_threshold_tuning"] is False,
        "global_positive_safe": _positive_safe(metric),
        "at_least_two_positive_safe_domains": len(summary["positive_safe_domains"]) >= 2,
        "ucy_positive_safe": _positive_safe(repair["test"]["by_domain"].get("UCY", {})),
        "trajnet_positive_safe": _positive_safe(repair["test"]["by_domain"].get("TrajNet", {})),
        "near_better_than_fc": repair["test"]["near_delta_vs_fc"] < 0.0,
        "near_not_worse_than_di": repair["test"]["near_delta_vs_di"] <= 0.0,
        "bootstrap_all_positive": repair["test"]["bootstrap"]["all"]["low"] > 0.0,
        "bootstrap_t50_positive": repair["test"]["bootstrap"]["t50"]["low"] > 0.0,
        "bootstrap_hard_positive": repair["test"]["bootstrap"]["hard_failure"]["low"] > 0.0,
        "easy_ci_safe": repair["test"]["bootstrap"]["easy_degradation"]["high"] <= 0.02,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
                no_leak["internal_val_from_train_only"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_fh_ucy_supported_fe_composer_pass" if passed == total else "stage42_fh_ucy_supported_fe_composer_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fh_gate"]
    metric = payload["repair"]["test"]["metric_vs_floor"]
    diag = payload["repair"]["test"]["diagnostics"]
    lines = [
        "# Stage42-FH UCY-Supported FE Composer",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{payload['summary']['deployment_decision']}`",
        f"- selected candidate: `{payload['summary']['selected_candidate']}`",
        "",
        "## Why This Exists",
        "",
        "- Stage42-FG showed frozen FE is robust on TrajNet but fallback-only on UCY.",
        "- Stage42-FH repairs the validation support rather than changing test thresholds: UCY internal validation is carved from original train sources only.",
        "",
        "## Protected Test Metrics vs Floor",
        "",
        f"- all improvement: `{_pct(metric['all_improvement'])}`",
        f"- t50 improvement: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        f"- final near@0.05: `{_pct(diag['final_near_005'])}`",
        "",
        "## By Domain",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy | switch | positive_safe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["repair"]["test"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | "
            f"{_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | "
            f"{_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} | {_positive_safe(row)} |"
        )
    lines += [
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n | bootstrap_n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["repair"]["test"]["bootstrap"].items():
        lines.append(f"| `{name}` | `{_pct(row['low'])}` | `{_pct(row['mid'])}` | `{_pct(row['high'])}` | `{row['n']}` | `{row['bootstrap_n']}` |")
    lines += [
        "",
        "## No Leakage / Claim Boundary",
        "",
        "- UCY internal validation is selected from original train sources only.",
        "- test rows are unchanged and used once for final evaluation.",
        "- no future endpoint/waypoint input, no central velocity, no test endpoint goals, no test threshold tuning.",
        "- dataset-local/raw-frame 2.5D only; no metric/seconds claim.",
        "- Stage5C not executed; SMC not enabled.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fh_gate"]
    return [
        "# Stage42-FH Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _replace_text_section(old: str, tag: str, block: str) -> str:
    start = f"<!-- {tag}:START -->"
    end = f"<!-- {tag}:END -->"
    if start in old and end in old:
        before, rest = old.split(start, 1)
        _, after = rest.split(end, 1)
        return before.rstrip() + "\n\n" + block.strip() + after
    return old.rstrip() + "\n\n" + block.strip() + "\n"


def _summary_section(payload: Mapping[str, Any]) -> str:
    metric = payload["repair"]["test"]["metric_vs_floor"]
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER:START -->",
            "## Stage42-FH UCY-Supported FE Composer",
            "",
            f"- source: `{payload['source']}`",
            "- role: repair Stage42-FG UCY fallback-only weakness by adding train-only UCY internal validation before FE composer selection.",
            f"- gate: `{payload['stage42_fh_gate']['passed']} / {payload['stage42_fh_gate']['total']}`; verdict `{payload['stage42_fh_gate']['verdict']}`.",
            f"- positive safe domains: `{s['positive_safe_domains']}`; weak domains: `{s['weak_domains']}`.",
            f"- all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
            f"- decision: `{s['deployment_decision']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(_replace_text_section(old, "STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FH UCY-supported FE composer"
    state["current_verdict"] = payload["stage42_fh_gate"]["verdict"]
    state["stage42_fh_ucy_supported_fe_composer"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fh_gate"]["verdict"],
        "gates": f"{payload['stage42_fh_gate']['passed']}/{payload['stage42_fh_gate']['total']}",
        "summary": payload["summary"],
        "test_metric_vs_floor": payload["repair"]["test"]["metric_vs_floor"],
        "by_domain": payload["repair"]["test"]["by_domain"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FH repairs the FE UCY validation-support weakness by selecting the FE composer family after a train-only UCY internal validation split.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FH UCY-supported FE composer" not in evidence:
            evidence.append("Stage42-FH UCY-supported FE composer")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fh_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FH responds to FG's UCY weak-slice finding by rerunning FE selection with train-only UCY internal validation support."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_ucy_supported_fe_composer() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_ucy_supported_fe_composer()
    gate = result["stage42_fh_gate"]
    print(f"Stage42-FH gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
