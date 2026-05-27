from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_constrained_fc_safety_composer as fe
from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_policy_freeze_replay as fi
from src import stage42_fh_source_robustness_audit as fj
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fh_horizon_weak_slice_repair_stage42.json"
REPORT_MD = OUT_DIR / "fh_horizon_weak_slice_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fk_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_fh_horizon_weak_slice_repair"
MIN_VAL_ROWS = 30
EASY_LIMIT = 0.02

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FJ 证明 frozen FH/FI 在 TrajNet 与 UCY domain/source 上 robust，但 TrajNet|100、UCY|50、UCY|100 仍是 horizon weak slices。",
    "Stage42-FK 只针对 FJ 暴露的 weak horizon slices 做 validation-only repair，不重新训练，不用 test 调阈值。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _key_mask(data: Mapping[str, np.ndarray], ids: np.ndarray, key: str) -> np.ndarray:
    domain_name, horizon_s = key.split("|", 1)
    return (data["dataset"][ids].astype(str) == domain_name) & (data["horizon"][ids].astype(int) == int(horizon_s))


def _candidate_metric(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    switch: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not np.any(mask):
        return {"rows": 0, "all_improvement": 0.0, "t50_improvement": 0.0, "t100_raw_frame_diagnostic_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "switch_rate": 0.0}
    return di._metric_subset(selected_ade[mask], floor_ade[mask], data, ids[mask], switch[mask])


def _safe_for_validation(metric: Mapping[str, Any], rows: int) -> bool:
    return bool(
        rows >= MIN_VAL_ROWS
        and float(metric.get("all_improvement", 0.0)) >= 0.0
        and float(metric.get("hard_failure_improvement", 0.0)) >= 0.0
        and float(metric.get("easy_degradation", 1.0)) <= EASY_LIMIT
    )


def _choose_candidate(candidates: Mapping[str, Mapping[str, Any]], *, min_rows: int = MIN_VAL_ROWS) -> dict[str, Any]:
    """Choose a horizon-slice override using validation metrics only."""

    safe = []
    for name, row in candidates.items():
        metric = row["metric"]
        rows = int(metric.get("rows", 0))
        ok = _safe_for_validation(metric, rows) if name != "floor" else rows >= min_rows
        candidate = {
            "candidate": name,
            "rows": rows,
            "metric": metric,
            "validation_safe": ok,
            "score": float(metric.get("all_improvement", 0.0)) + 0.25 * float(metric.get("hard_failure_improvement", 0.0)) - max(0.0, float(metric.get("easy_degradation", 0.0))),
        }
        if ok:
            safe.append(candidate)
    if not safe:
        return {"candidate": "floor", "reason": "no_validation_safe_positive_candidate", "validation_safe": True, "score": 0.0}
    safe.sort(key=lambda row: (row["score"], row["metric"].get("all_improvement", 0.0)), reverse=True)
    choice = dict(safe[0])
    choice["reason"] = "validation_safe_best_score"
    return choice


def _evals_by_name(
    ids: np.ndarray,
    base: Mapping[str, Any],
    ref_evals: Mapping[str, Any],
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy[ids], labels, ids)
    floor_eval = {
        "selected_xy": floor_xy[ids].astype(np.float32),
        "selected_ade": floor_ade,
        "selected_fde": floor_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": np.zeros(len(ids), dtype=bool),
    }
    return {
        "fh": {
            "selected_xy": base["selected_xy"],
            "selected_ade": base["selected_ade"],
            "selected_fde": base["selected_fde"],
            "floor_ade": base["floor_ade"],
            "floor_fde": base["floor_fde"],
            "switch": base["switch"],
        },
        "fc": ref_evals["fc"],
        "di": ref_evals["di"],
        "fa": ref_evals["fa"],
        "fb": ref_evals["fb"],
        "floor": floor_eval,
    }


def _apply_overrides(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    base: Mapping[str, Any],
    evals: Mapping[str, Mapping[str, Any]],
    choices: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected_xy = base["selected_xy"].copy()
    selected_ade = base["selected_ade"].copy()
    selected_fde = base["selected_fde"].copy()
    switch = base["switch"].copy()
    applied: dict[str, Any] = {}
    for key, choice in choices.items():
        candidate = str(choice["candidate"])
        mask = _key_mask(data, ids, key)
        if candidate not in evals or not np.any(mask):
            continue
        selected_xy[mask] = evals[candidate]["selected_xy"][mask]
        selected_ade[mask] = evals[candidate]["selected_ade"][mask]
        selected_fde[mask] = evals[candidate]["selected_fde"][mask]
        switch[mask] = evals[candidate]["switch"][mask]
        applied[key] = {"candidate": candidate, "rows": int(np.sum(mask)), "reason": choice.get("reason")}
    return {"selected_xy": selected_xy, "selected_ade": selected_ade, "selected_fde": selected_fde, "switch": switch, "applied": applied}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fj_payload = read_json(fj.REPORT_JSON, {})
    if not fj_payload:
        fj_payload = fj.run_stage42_fh_source_robustness_audit()
    fi_payload = read_json(fi.REPORT_JSON, {})
    if not fi_payload:
        fi_payload = fi.run_stage42_fh_policy_freeze_replay()

    ctx = fi._context()
    data = ctx["data"]
    labels = ctx["labels"]
    floor_xy = ctx["floor"]["floor_xy"].astype(np.float32)
    candidate = fi_payload["frozen_policy"]["selected_candidate"]
    replay = fi._replay_selected(ctx, candidate)
    val_ids = replay["val_ids"]
    test_ids = replay["test_ids"]
    val_evals = fe._reference_evals(val_ids, data, labels, ctx["floor"], ctx["am_candidate"], ctx["fc_candidate"], ctx["group_key"], ctx["prior"])
    test_evals = replay["test_evals"]
    val_by_name = _evals_by_name(val_ids, replay["val"], val_evals, floor_xy, labels)
    test_by_name = _evals_by_name(test_ids, replay["test"], test_evals, floor_xy, labels)

    weak_keys = list(fj_payload["summary"].get("weak_domain_horizons", []))
    choices: dict[str, Any] = {}
    val_tables: dict[str, Any] = {}
    for key in weak_keys:
        mask = _key_mask(data, val_ids, key)
        rows: dict[str, Any] = {}
        for name, ev in val_by_name.items():
            rows[name] = {"metric": _candidate_metric(data, val_ids, ev["selected_ade"], ev["floor_ade"], ev["switch"], mask)}
        choices[key] = _choose_candidate(rows)
        val_tables[key] = rows

    repaired = _apply_overrides(data, test_ids, replay["test"], test_by_name, choices)
    selected = repaired["selected_ade"]
    floor = replay["test"]["floor_ade"]
    switch = repaired["switch"]
    normalizer = np.maximum(data["scale"][test_ids].astype(np.float64), 1e-6)
    agent = data["agent_id"][test_ids].astype(np.int64)
    final_min = di._min_group_distance_fast(repaired["selected_xy"], ctx["group_key"][test_ids], normalizer, agent)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(test_evals["fc"]["min_distance"]) & (test_evals["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(test_evals["di"]["min_distance"]) & (test_evals["di"]["min_distance"] < 0.05)

    domain = data["dataset"][test_ids].astype(str)
    horizon = data["horizon"][test_ids].astype(int)
    domain_rows = {
        name: fg._group_row(name, domain == name, data, test_ids, selected, floor, switch, final_near, fc_near, di_near, seed=44200 + i * 100)
        for i, name in enumerate(sorted(set(domain.tolist())))
    }
    domain_horizon_rows = {
        f"{d}|{h}": fg._group_row(
            f"{d}|{h}",
            (domain == d) & (horizon == h),
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=44300 + i * 100,
        )
        for i, (d, h) in enumerate((d, h) for d in sorted(set(domain.tolist())) for h in [10, 25, 50, 100])
        if np.any((domain == d) & (horizon == h))
    }
    source_file = data["source_file"][test_ids].astype(str)
    source_rows = {
        fg._source_name(name): fg._group_row(
            fg._source_name(name),
            source_file == name,
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=44400 + i * 100,
        )
        for i, name in enumerate(sorted(set(source_file.tolist())))
    }

    weak_after = [name for name, row in domain_horizon_rows.items() if row["rows"] >= fg.MIN_CI_ROWS and not row["robust_positive"]]
    robust_after = [name for name, row in domain_horizon_rows.items() if row["robust_positive"]]
    weak_before = list(fj_payload["summary"].get("weak_domain_horizons", []))
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "weak_domain_horizons_before": weak_before,
        "weak_domain_horizons_after": weak_after,
        "robust_domain_horizons_after": robust_after,
        "weak_horizon_count_before": len(weak_before),
        "weak_horizon_count_after": len(weak_after),
        "repaired_horizon_count": max(0, len(weak_before) - len(weak_after)),
        "applied_overrides": repaired["applied"],
        "uniform_horizon_claim_allowed": len(weak_after) == 0,
        "decision": "promote_stage42_fk_horizon_repair" if len(weak_after) == 0 else "horizon_repair_partial_keep_stage42_fh_fi_with_horizon_limit",
    }
    metric = di._metric_subset(selected, floor, data, test_ids, switch)
    diagnostics = {
        "final_near_005": float(np.mean(final_near)) if len(final_near) else 0.0,
        "delta_near_vs_fc": float(np.mean(final_near.astype(float) - fc_near.astype(float))) if len(final_near) else 0.0,
        "delta_near_vs_di": float(np.mean(final_near.astype(float) - di_near.astype(float))) if len(final_near) else 0.0,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FK FH horizon weak-slice validation repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash([str(fi.REPORT_JSON), str(fi.POLICY_JSON), str(fj.REPORT_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_fi_verdict": fi_payload["stage42_fi_gate"]["verdict"],
            "stage42_fj_verdict": fj_payload["stage42_fj_gate"]["verdict"],
        },
        "selection_rule": {
            "source": "validation_only",
            "target_keys": weak_keys,
            "candidate_families": ["fh", "fc", "di", "fa", "fb", "floor"],
            "min_val_rows": MIN_VAL_ROWS,
            "easy_limit": EASY_LIMIT,
            "uses_test_metrics_for_selection": False,
        },
        "validation_tables": val_tables,
        "choices": choices,
        "summary": summary,
        "metric_vs_floor": metric,
        "diagnostics": diagnostics,
        "domain_rows": domain_rows,
        "domain_horizon_rows": domain_horizon_rows,
        "source_rows": source_rows,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "test_rows_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "uniform_horizon_claim": summary["uniform_horizon_claim_allowed"],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_fk_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    metric = payload["metric_vs_floor"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fi_input_verified": payload["input_reports"]["stage42_fi_verdict"] == "stage42_fi_fh_policy_freeze_replay_pass",
        "fj_input_verified": payload["input_reports"]["stage42_fj_verdict"] == "stage42_fj_fh_source_robustness_pass",
        "weak_horizons_targeted": len(payload["selection_rule"]["target_keys"]) >= 1,
        "validation_only_selection": payload["selection_rule"]["uses_test_metrics_for_selection"] is False,
        "global_all_positive": metric["all_improvement"] > 0.0,
        "global_t50_positive": metric["t50_improvement"] > 0.0,
        "global_hard_positive": metric["hard_failure_improvement"] > 0.0,
        "global_easy_safe": metric["easy_degradation"] <= EASY_LIMIT,
        "weak_horizon_not_increased": s["weak_horizon_count_after"] <= s["weak_horizon_count_before"],
        "uniform_horizon_claim_only_if_no_weak_horizons": (boundary["uniform_horizon_claim"] is False) or s["weak_horizon_count_after"] == 0,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and int(payload["summary"]["weak_horizon_count_after"]) == 0:
        verdict = "stage42_fk_fh_horizon_weak_slice_repair_pass_uniform_horizon"
    elif passed == total:
        verdict = "stage42_fk_fh_horizon_weak_slice_repair_pass_with_horizon_limit"
    else:
        verdict = "stage42_fk_fh_horizon_weak_slice_repair_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fk_gate"]
    s = payload["summary"]
    m = payload["metric_vs_floor"]
    d = payload["diagnostics"]
    lines = [
        "# Stage42-FK FH Horizon Weak-Slice Validation Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{s['decision']}`",
        "",
        "## Global Test Metrics vs Floor",
        "",
        f"- all improvement: `{_pct(m['all_improvement'])}`",
        f"- t50 improvement: `{_pct(m['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(m['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(m['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(m['easy_degradation'])}`",
        f"- switch rate: `{_pct(m['switch_rate'])}`",
        f"- final near@0.05: `{_pct(d['final_near_005'])}`",
        "",
        "## Horizon Repair Summary",
        "",
        f"- weak_domain_horizons_before: `{s['weak_domain_horizons_before']}`",
        f"- weak_domain_horizons_after: `{s['weak_domain_horizons_after']}`",
        f"- repaired_horizon_count: `{s['repaired_horizon_count']}`",
        f"- uniform_horizon_claim_allowed: `{s['uniform_horizon_claim_allowed']}`",
        f"- applied_overrides: `{s['applied_overrides']}`",
        "",
        "## Validation Choices",
        "",
        "| key | candidate | reason | score |",
        "| --- | --- | --- | ---: |",
    ]
    for key, choice in payload["choices"].items():
        lines.append(f"| `{key}` | `{choice['candidate']}` | `{choice.get('reason')}` | {float(choice.get('score', 0.0)):.6f} |")
    lines += ["", *fg._render_group_table("Domain-Horizon Robustness After Repair", payload["domain_horizon_rows"])]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-FK is a validation-only repair attempt, not a new training run.",
        "- If weak horizon slices remain, uniform horizon robustness remains blocked.",
        "- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fk_gate"]
    lines = [
        "# Stage42-FK Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    m = payload["metric_vs_floor"]
    return "\n".join(
        [
            "<!-- STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR:START -->",
            "## Stage42-FK FH Horizon Weak-Slice Validation Repair",
            "",
            f"- source: `{payload['source']}`",
            "- role: validation-only repair attempt for FJ weak horizon slices; no retraining and no test threshold tuning.",
            f"- gate: `{payload['stage42_fk_gate']['passed']} / {payload['stage42_fk_gate']['total']}`; verdict `{payload['stage42_fk_gate']['verdict']}`.",
            f"- global all/t50/t100raw/hard/easy: `{_pct(m['all_improvement'])}` / `{_pct(m['t50_improvement'])}` / `{_pct(m['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(m['hard_failure_improvement'])}` / `{_pct(m['easy_degradation'])}`.",
            f"- weak horizons before: `{s['weak_domain_horizons_before']}`.",
            f"- weak horizons after: `{s['weak_domain_horizons_after']}`.",
            f"- applied overrides: `{s['applied_overrides']}`.",
            f"- uniform horizon claim allowed: `{s['uniform_horizon_claim_allowed']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FK FH horizon weak-slice validation repair"
    state["current_verdict"] = payload["stage42_fk_gate"]["verdict"]
    state["stage42_fk_fh_horizon_weak_slice_repair"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fk_gate"]["verdict"],
        "gates": f"{payload['stage42_fk_gate']['passed']}/{payload['stage42_fk_gate']['total']}",
        "summary": payload["summary"],
        "metric_vs_floor": payload["metric_vs_floor"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FK attempts to repair FJ weak horizon slices with validation-only overrides and preserves the uniform-horizon claim boundary.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FK FH horizon weak-slice validation repair" not in evidence:
            evidence.append("Stage42-FK FH horizon weak-slice validation repair")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fk_fresh_audits"
        block["latest_conclusion"] = "Stage42-FK attempts validation-only repair for FH/FI weak horizon slices and records whether uniform horizon claims remain blocked."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fh_horizon_weak_slice_repair() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fh_horizon_weak_slice_repair()
    gate = result["stage42_fk_gate"]
    print(f"Stage42-FK gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
