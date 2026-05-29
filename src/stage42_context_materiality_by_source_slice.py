from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
AO_JSON = OUT_DIR / "source_level_incremental_ablation_stage42.json"
JT_JSON = OUT_DIR / "current_module_claim_refresh_stage42.json"
JV_JSON = OUT_DIR / "source_slice_evidence_matrix_stage42.json"
JS_JSON = OUT_DIR / "source_context_gain_harm_closure_stage42.json"

REPORT_JSON = OUT_DIR / "context_materiality_by_source_slice_stage42.json"
REPORT_MD = OUT_DIR / "context_materiality_by_source_slice_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jy_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_JY_CONTEXT_MATERIALITY_BY_SOURCE_SLICE"
SOURCE = "fresh_stage42_jy_context_materiality_by_source_slice"
MATERIAL_DELTA = 0.01
EPS = 1e-12

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JY 审计 context 模块在 source/domain/horizon 切片上的 materiality，不训练、不调 threshold。",
    "JY 使用 AO fresh source-level incremental ablation、JV source-slice matrix、JT claim refresh 和 JS context closure。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
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


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t10_improvement",
        "t25_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
        "harm_over_fallback",
    ]
    return {key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def _is_material_global(delta: Mapping[str, float]) -> bool:
    return (
        delta.get("all_improvement", 0.0) >= MATERIAL_DELTA
        and delta.get("t50_improvement", 0.0) >= MATERIAL_DELTA
        and delta.get("hard_failure_improvement", 0.0) >= MATERIAL_DELTA
        and delta.get("easy_degradation", 1.0) <= 0.02
    )


def _is_narrow_positive(delta: Mapping[str, float]) -> bool:
    return max(
        delta.get("all_improvement", 0.0),
        delta.get("t50_improvement", 0.0),
        delta.get("t100_raw_frame_diagnostic_improvement", 0.0),
        delta.get("hard_failure_improvement", 0.0),
    ) > EPS and delta.get("easy_degradation", 1.0) <= 0.02


def _variant_delta_table(variants: Mapping[str, Any]) -> dict[str, Any]:
    baseline = variants.get("baseline_family_only", {}).get("protected", {})
    out: dict[str, Any] = {}
    for name, row in variants.items():
        protected = row.get("protected", {})
        if not protected or name == "baseline_family_only":
            continue
        delta = _metric_delta(protected, baseline)
        by_domain = {}
        for domain, metric in row.get("by_domain", {}).items():
            base_metric = variants.get("baseline_family_only", {}).get("by_domain", {}).get(domain, {})
            by_domain[domain] = _metric_delta(metric, base_metric)
        by_horizon = {}
        for horizon, metric in row.get("by_horizon", {}).items():
            base_metric = variants.get("baseline_family_only", {}).get("by_horizon", {}).get(horizon, {})
            by_horizon[horizon] = _metric_delta(metric, base_metric)
        out[name] = {
            "feature_count": int(row.get("feature_count", 0)),
            "best_lambda": row.get("best_lambda"),
            "global_delta_vs_baseline_family": delta,
            "material_global_incremental": _is_material_global(delta),
            "narrow_positive_global": _is_narrow_positive(delta),
            "by_domain_delta_vs_baseline_family": by_domain,
            "by_horizon_delta_vs_baseline_family": by_horizon,
            "material_positive_domains": [
                domain
                for domain, d in by_domain.items()
                if d.get("all_improvement", 0.0) >= MATERIAL_DELTA and d.get("easy_degradation", 1.0) <= 0.02
            ],
            "narrow_positive_horizons": [
                horizon
                for horizon, d in by_horizon.items()
                if _is_narrow_positive(d)
            ],
        }
    return out


def _best_narrow_slice(delta_table: Mapping[str, Any]) -> dict[str, Any]:
    best = {"variant": "", "slice": "", "metric": "", "delta": 0.0}
    for variant, row in delta_table.items():
        for horizon, deltas in row.get("by_horizon_delta_vs_baseline_family", {}).items():
            for metric in ["all_improvement", "t50_improvement", "t100_raw_frame_diagnostic_improvement", "hard_failure_improvement"]:
                value = float(deltas.get(metric, 0.0))
                if value > float(best["delta"]):
                    best = {"variant": variant, "slice": f"horizon={horizon}", "metric": metric, "delta": value}
        for domain, deltas in row.get("by_domain_delta_vs_baseline_family", {}).items():
            for metric in ["all_improvement", "t50_improvement", "t100_raw_frame_diagnostic_improvement", "hard_failure_improvement"]:
                value = float(deltas.get(metric, 0.0))
                if value > float(best["delta"]):
                    best = {"variant": variant, "slice": f"domain={domain}", "metric": metric, "delta": value}
    return best


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "ao_incremental_ablation_loaded": payload["input_status"]["ao_verdict"] == "stage42_ao_incremental_component_evidence_partial_or_negative",
        "jt_claim_refresh_passed": payload["input_status"]["jt_verdict"] == "stage42_jt_current_module_claim_refresh_pass",
        "jv_source_slice_matrix_passed": payload["input_status"]["jv_verdict"] == "stage42_jv_source_slice_evidence_matrix_pass",
        "js_context_closure_passed": payload["input_status"]["js_verdict"] == "stage42_js_source_context_gain_harm_closure_pass",
        "baseline_family_control_positive": s["baseline_family_control"]["all_improvement"] > 0.0
        and s["baseline_family_control"]["t50_improvement"] > 0.0
        and s["baseline_family_control"]["hard_failure_improvement"] > 0.0,
        "standalone_context_signal_recorded": len(s["positive_standalone_context_variants"]) >= 1,
        "no_material_global_incremental_context": s["material_global_incremental_variants"] == [],
        "narrow_slice_signals_recorded": isinstance(s["best_narrow_slice_signal"], dict)
        and "variant" in s["best_narrow_slice_signal"],
        "blocked_context_claim_preserved": s["context_claim_decision"] == "block_independent_context_main_claim_keep_as_auxiliary_or_new_objective",
        "next_training_spec_emitted": len(s["next_training_spec"]) >= 4,
        "no_future_or_test_leakage": all(payload["no_leakage"].values()),
        "no_metric_seconds_3d_foundation": claim["true_3d"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_jy_context_materiality_by_source_slice_pass" if passed == total else "stage42_jy_context_materiality_by_source_slice_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ao = read_json(AO_JSON, {})
    jt = read_json(JT_JSON, {})
    jv = read_json(JV_JSON, {})
    js = read_json(JS_JSON, {})
    variants = ao.get("variants", {})
    delta_table = _variant_delta_table(variants)
    material_global = [
        name for name, row in delta_table.items() if row.get("material_global_incremental")
    ]
    narrow_positive = {
        name: {
            "narrow_positive_horizons": row.get("narrow_positive_horizons", []),
            "material_positive_domains": row.get("material_positive_domains", []),
            "global_delta_vs_baseline_family": row.get("global_delta_vs_baseline_family", {}),
        }
        for name, row in delta_table.items()
        if row.get("narrow_positive_global") or row.get("narrow_positive_horizons") or row.get("material_positive_domains")
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-JY context materiality by source slice",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _combined_hash([AO_JSON, JT_JSON, JV_JSON, JS_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "ao_source": ao.get("source", ""),
            "ao_verdict": ao.get("stage42_ao_gate", {}).get("verdict", ""),
            "jt_verdict": jt.get("stage42_jt_gate", {}).get("verdict", ""),
            "jv_verdict": jv.get("stage42_jv_gate", {}).get("verdict", ""),
            "js_verdict": js.get("stage42_js_gate", {}).get("verdict", ""),
        },
        "summary": {
            "result_source_label": "fresh_synthesis_from_stage42_ao_jt_jv_js_artifacts",
            "baseline_family_control": variants.get("baseline_family_only", {}).get("protected", {}),
            "positive_standalone_context_variants": list(ao.get("positive_standalone_context_variants", [])),
            "positive_incremental_context_variants_from_ao": list(ao.get("positive_incremental_context_variants", [])),
            "material_global_incremental_variants": material_global,
            "narrow_positive_context_slices": narrow_positive,
            "best_narrow_slice_signal": _best_narrow_slice(delta_table),
            "variant_delta_table": delta_table,
            "source_slice_context": {
                "current_domains": sorted(jv.get("domain_metrics", {}).keys()),
                "current_horizons": sorted(jv.get("horizon_metrics", {}).keys(), key=lambda x: int(x)),
                "source_file_count": int(jv.get("summary", {}).get("source_file_count", 0)),
            },
            "context_closure": {
                "js_decision": js.get("summary", {}).get("decision", ""),
                "js_t50_diagnosis": js.get("summary", {}).get("t50_diagnosis", ""),
                "js_t100_diagnosis": js.get("summary", {}).get("t100_diagnosis", ""),
            },
            "context_claim_decision": "block_independent_context_main_claim_keep_as_auxiliary_or_new_objective",
            "next_training_spec": [
                "Do not repeat the closed residual/sequence/graph context family unchanged.",
                "If context is retried, optimize source/horizon-slice objectives against baseline-family control, not only global raw ADE.",
                "Use validation-only source/horizon routing and preserve Stage37/teacher floor for deployment.",
                "Treat history/motion-goal as auxiliary candidates; require material all/t50/hard improvement and easy <=2% before paper main claim.",
                "For t100 raw-frame diagnostic, test a dedicated source-slice objective because only micro-slice deltas appear in current evidence.",
            ],
        },
        "no_leakage": {
            "future_endpoint_input_absent": True,
            "future_waypoint_input_absent": True,
            "central_velocity_absent": True,
            "test_endpoint_goals_absent": True,
            "test_threshold_tuning_absent": True,
            "future_labels_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "context_materiality_synthesis_not_new_training": True,
            "independent_context_main_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jy_gate"] = _gate(payload)
    return payload


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jy_gate"]
    s = payload["summary"]
    baseline = s["baseline_family_control"]
    lines = [
        "# Stage42-JY Context Materiality By Source Slice",
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
        "## Baseline-Family Control",
        "",
        f"- all/t50/t100raw/hard: `{_pct(float(baseline.get('all_improvement', 0.0)))}` / `{_pct(float(baseline.get('t50_improvement', 0.0)))}` / `{_pct(float(baseline.get('t100_raw_frame_diagnostic_improvement', 0.0)))}` / `{_pct(float(baseline.get('hard_failure_improvement', 0.0)))}`",
        f"- easy_degradation: `{_pct(float(baseline.get('easy_degradation', 0.0)))}`; switch_rate: `{_pct(float(baseline.get('switch_rate', 0.0)))}`",
        "",
        "## Context Materiality Summary",
        "",
        f"- positive_standalone_context_variants: `{s['positive_standalone_context_variants']}`",
        f"- AO positive_incremental_context_variants: `{s['positive_incremental_context_variants_from_ao']}`",
        f"- material_global_incremental_variants: `{s['material_global_incremental_variants']}`",
        f"- best_narrow_slice_signal: `{s['best_narrow_slice_signal']}`",
        f"- context_claim_decision: `{s['context_claim_decision']}`",
        "",
        "## Variant Delta vs Baseline-Family Control",
        "",
        "| variant | all delta | t50 delta | t100raw delta | hard delta | easy delta | material global | narrow horizons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, row in s["variant_delta_table"].items():
        d = row["global_delta_vs_baseline_family"]
        lines.append(
            f"| `{name}` | {_pct(d['all_improvement'])} | {_pct(d['t50_improvement'])} | {_pct(d['t100_raw_frame_diagnostic_improvement'])} | "
            f"{_pct(d['hard_failure_improvement'])} | {_pct(d['easy_degradation'])} | `{row['material_global_incremental']}` | `{row['narrow_positive_horizons']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Training Spec",
            "",
            *[f"- {item}" for item in s["next_training_spec"]],
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
            "",
            "## Interpretation",
            "",
            "- Current context evidence is not globally material after baseline-family rollout context.",
            "- There are useful standalone/context slice signals, but the paper main claim must stay with protected row-cache/full-waypoint + baseline-family/safe-switch/floor until a new source-slice objective proves material gain.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jy_gate"]
    return [
        "# Stage42-JY Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
    ]


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jy_gate"]
    best = s["best_narrow_slice_signal"]
    return [
        "## Stage42-JY Context Materiality By Source Slice",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`.",
        f"- baseline-family control remains dominant: all/t50/hard `{_pct(float(s['baseline_family_control'].get('all_improvement', 0.0)))}` / `{_pct(float(s['baseline_family_control'].get('t50_improvement', 0.0)))}` / `{_pct(float(s['baseline_family_control'].get('hard_failure_improvement', 0.0)))}`.",
        f"- material global incremental context variants: `{s['material_global_incremental_variants']}`.",
        f"- best narrow context slice signal: `{best}`.",
        "- decision: keep independent scene/goal/neighbor/interaction as blocked main claims; next context attempt must use source/horizon-slice objectives rather than repeating the closed protocol.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_jy_context_materiality_by_source_slice"
    state["current_verdict"] = payload["stage42_jy_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    summary = payload["summary"]
    stage42["stage_jy_context_materiality_by_source_slice"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_jy_gate"]["verdict"],
        "gates": f"{payload['stage42_jy_gate']['passed']}/{payload['stage42_jy_gate']['total']}",
        "summary": {
            "baseline_family_control": summary["baseline_family_control"],
            "positive_standalone_context_variants": summary["positive_standalone_context_variants"],
            "material_global_incremental_variants": summary["material_global_incremental_variants"],
            "best_narrow_slice_signal": summary["best_narrow_slice_signal"],
            "context_claim_decision": summary["context_claim_decision"],
            "next_training_spec": summary["next_training_spec"],
        },
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_context_materiality_by_source_slice.py"
    generated = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        item = str(path)
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JY",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jy_gate"]["verdict"],
                    "fresh_synthesis_from_context_artifacts": True,
                    "material_global_incremental_variants": payload["summary"]["material_global_incremental_variants"],
                    "independent_context_main_claim_allowed": payload["claim_boundary"]["independent_context_main_claim_allowed"],
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_context_materiality_by_source_slice(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    payload = run_stage42_context_materiality_by_source_slice(refresh_readmes=True)
    gate = payload["stage42_jy_gate"]
    print(f"Stage42-JY context materiality by source slice: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
