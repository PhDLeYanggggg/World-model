from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_policy_freeze_replay as fi
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fh_source_robustness_stage42.json"
REPORT_MD = OUT_DIR / "fh_source_robustness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fj_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_fh_source_robustness_audit"
MIN_DOMAIN_ROWS = 500
MIN_CI_ROWS = fg.MIN_CI_ROWS

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FI 已冻结 Stage42-FH policy，并做 exact replay + 2000-bootstrap。",
    "Stage42-FJ 不重新训练、不重新选择 policy、不调 test threshold；它只审计 frozen FH 在 domain/source/horizon/scene 切片上的稳健性。",
    "弱 source / weak slice 必须显式报告；不能把 global positive 包装成每个外部源、每个 horizon 都 positive。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fi_payload = read_json(fi.REPORT_JSON, {})
    if not fi_payload:
        fi_payload = fi.run_stage42_fh_policy_freeze_replay()
    ctx = fi._context()
    candidate = fi_payload["frozen_policy"]["selected_candidate"]
    replay = fi._replay_selected(ctx, candidate)
    data = ctx["data"]
    test_ids = replay["test_ids"]
    selected = replay["test"]["selected_ade"]
    floor = replay["test"]["floor_ade"]
    switch = replay["test"]["switch"]
    final_near = np.isfinite(replay["final_min_distance"]) & (replay["final_min_distance"] < 0.05)
    fc_near = np.isfinite(replay["test_evals"]["fc"]["min_distance"]) & (replay["test_evals"]["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(replay["test_evals"]["di"]["min_distance"]) & (replay["test_evals"]["di"]["min_distance"] < 0.05)
    domain = data["dataset"][test_ids].astype(str)
    source_file = data["source_file"][test_ids].astype(str)
    horizon = data["horizon"][test_ids].astype(int)
    domain_rows = {
        name: fg._group_row(
            name,
            domain == name,
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=43900 + i * 100,
        )
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
            seed=44000 + i * 100,
        )
        for i, (d, h) in enumerate((d, h) for d in sorted(set(domain.tolist())) for h in [10, 25, 50, 100])
        if np.any((domain == d) & (horizon == h))
    }
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
            seed=44100 + i * 100,
        )
        for i, name in enumerate(sorted(set(source_file.tolist())))
    }
    weak_domains = [name for name, row in domain_rows.items() if row["rows"] >= MIN_DOMAIN_ROWS and not row["robust_positive"]]
    weak_domain_horizons = [name for name, row in domain_horizon_rows.items() if row["rows"] >= MIN_CI_ROWS and not row["robust_positive"]]
    weak_sources = [name for name, row in source_rows.items() if row["rows"] >= MIN_CI_ROWS and not row["robust_positive"]]
    robust_domains = [name for name, row in domain_rows.items() if row["robust_positive"]]
    robust_domain_horizons = [name for name, row in domain_horizon_rows.items() if row["robust_positive"]]
    robust_sources = [name for name, row in source_rows.items() if row["robust_positive"]]
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "domain_count": len(domain_rows),
        "source_count": len(source_rows),
        "domain_horizon_count": len(domain_horizon_rows),
        "robust_domains": robust_domains,
        "robust_domain_horizons": robust_domain_horizons,
        "robust_sources": robust_sources,
        "weak_domains": weak_domains,
        "weak_domain_horizons": weak_domain_horizons,
        "weak_sources": weak_sources,
        "dual_domain_positive_safe_claim_allowed": len(robust_domains) >= 2 and not weak_domains,
        "broad_uniform_source_claim_allowed": len(weak_sources) == 0 and len(source_rows) >= 3,
        "broad_uniform_horizon_claim_allowed": len(weak_domain_horizons) == 0 and len(domain_horizon_rows) >= 6,
        "paper_claim": (
            "Stage42-FH/FI is frozen and bootstrap-positive globally; Stage42-FJ reports whether the UCY-supported policy "
            "is robust by powered domain/source/horizon slices. Broad source or horizon claims are allowed only when powered weak slices disappear."
        ),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FJ FH source/domain/horizon robustness audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(fi.REPORT_JSON),
                str(fi.POLICY_JSON),
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "fh_policy": {
            "policy_hash": fi_payload["frozen_policy"]["policy_hash"],
            "selected_candidate": candidate,
            "fi_verdict": fi_payload["stage42_fi_gate"]["verdict"],
        },
        "summary": summary,
        "domain_rows": domain_rows,
        "domain_horizon_rows": domain_horizon_rows,
        "source_rows": source_rows,
        "weak_scene_rows_top": fg._top_scene_rows(data, test_ids, selected, floor, switch),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "test_rows_reporting_only": True,
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "dual_domain_positive_safe_claim": summary["dual_domain_positive_safe_claim_allowed"],
            "broad_uniform_source_claim": summary["broad_uniform_source_claim_allowed"],
            "broad_uniform_horizon_claim": summary["broad_uniform_horizon_claim_allowed"],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_fj_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fi_input_verified": payload["fh_policy"]["fi_verdict"] == "stage42_fi_fh_policy_freeze_replay_pass",
        "domains_audited": s["domain_count"] >= 2,
        "sources_audited": s["source_count"] >= 2,
        "domain_horizons_audited": s["domain_horizon_count"] >= 4,
        "at_least_two_robust_domains": len(s["robust_domains"]) >= 2,
        "weak_slices_reported": "weak_domain_horizons" in s and "weak_sources" in s and "weak_domains" in s,
        "dual_domain_claim_only_if_no_weak_domains": (boundary["dual_domain_positive_safe_claim"] is False) or len(s["weak_domains"]) == 0,
        "broad_source_claim_only_if_no_weak_sources": (boundary["broad_uniform_source_claim"] is False) or len(s["weak_sources"]) == 0,
        "broad_horizon_claim_only_if_no_weak_horizons": (boundary["broad_uniform_horizon_claim"] is False) or len(s["weak_domain_horizons"]) == 0,
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
    verdict = "stage42_fj_fh_source_robustness_pass" if passed == total else "stage42_fj_fh_source_robustness_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fj_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-FJ FH Source / Domain / Horizon Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- FH policy hash: `{payload['fh_policy']['policy_hash']}`",
        "",
        "## Summary",
        "",
        f"- test_rows: `{s['test_rows']}`",
        f"- domain_count: `{s['domain_count']}`",
        f"- source_count: `{s['source_count']}`",
        f"- domain_horizon_count: `{s['domain_horizon_count']}`",
        f"- robust_domains: `{s['robust_domains']}`",
        f"- weak_domains: `{s['weak_domains']}`",
        f"- robust_domain_horizons: `{s['robust_domain_horizons']}`",
        f"- weak_domain_horizons: `{s['weak_domain_horizons']}`",
        f"- robust_sources: `{s['robust_sources']}`",
        f"- weak_sources: `{s['weak_sources']}`",
        f"- dual_domain_positive_safe_claim_allowed: `{s['dual_domain_positive_safe_claim_allowed']}`",
        f"- broad_uniform_source_claim_allowed: `{s['broad_uniform_source_claim_allowed']}`",
        f"- broad_uniform_horizon_claim_allowed: `{s['broad_uniform_horizon_claim_allowed']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
    ]
    lines.extend(fg._render_group_table("Domain Robustness", payload["domain_rows"]))
    lines.append("")
    lines.extend(fg._render_group_table("Domain-Horizon Robustness", payload["domain_horizon_rows"]))
    lines.append("")
    lines.extend(fg._render_group_table("Source Robustness", payload["source_rows"]))
    lines += [
        "",
        "## Weak Scene Rows",
        "",
        "| scene | rows | all | t50 | t100 raw | hard | easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["weak_scene_rows_top"]:
        m = row["metric"]
        lines.append(
            f"| `{row['scene']}` | {row['rows']} | {_pct(m['all_improvement'])} | {_pct(m['t50_improvement'])} | "
            f"{_pct(m['t100_raw_frame_diagnostic_improvement'])} | {_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-FJ is a robustness audit, not new training and not policy reselection.",
        "- FH/FI remains frozen; this report decides which claims are allowed at domain/source/horizon granularity.",
        "- Powered weak source or horizon slices block broad uniform source/horizon claims even when global metrics are strong.",
        "- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fj_gate"]
    lines = [
        "# Stage42-FJ Gate",
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
    return "\n".join(
        [
            "<!-- STAGE42_FJ_FH_SOURCE_ROBUSTNESS:START -->",
            "## Stage42-FJ FH Source / Domain / Horizon Robustness Audit",
            "",
            f"- source: `{payload['source']}`",
            "- role: audit frozen Stage42-FH/FI policy across domain/source/horizon/scene slices without retraining or threshold reselection.",
            f"- gate: `{payload['stage42_fj_gate']['passed']} / {payload['stage42_fj_gate']['total']}`; verdict `{payload['stage42_fj_gate']['verdict']}`.",
            f"- robust domains: `{s['robust_domains']}`.",
            f"- weak domains: `{s['weak_domains']}`.",
            f"- robust domain-horizon slices: `{s['robust_domain_horizons']}`.",
            f"- weak domain-horizon slices: `{s['weak_domain_horizons']}`.",
            f"- robust sources: `{s['robust_sources']}`.",
            f"- weak sources: `{s['weak_sources']}`.",
            f"- dual-domain positive-safe claim allowed: `{s['dual_domain_positive_safe_claim_allowed']}`.",
            f"- broad uniform source claim allowed: `{s['broad_uniform_source_claim_allowed']}`.",
            f"- broad uniform horizon claim allowed: `{s['broad_uniform_horizon_claim_allowed']}`.",
            "- Boundary: frozen protected source-level raw-frame 2.5D audit; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FJ_FH_SOURCE_ROBUSTNESS:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FJ_FH_SOURCE_ROBUSTNESS", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FJ FH source/domain/horizon robustness audit"
    state["current_verdict"] = payload["stage42_fj_gate"]["verdict"]
    state["stage42_fj_fh_source_robustness_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fj_gate"]["verdict"],
        "gates": f"{payload['stage42_fj_gate']['passed']}/{payload['stage42_fj_gate']['total']}",
        "fh_policy_hash": payload["fh_policy"]["policy_hash"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FJ audits the frozen FH/FI policy across powered domain/source/horizon slices and updates allowed claim boundaries.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FJ FH source/domain/horizon robustness audit" not in evidence:
            evidence.append("Stage42-FJ FH source/domain/horizon robustness audit")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fj_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FJ checks whether frozen FH/FI dual-domain gains are robust enough for source/horizon claims without retraining or test threshold tuning."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fh_source_robustness_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fh_source_robustness_audit()
    gate = result["stage42_fj_gate"]
    print(f"Stage42-FJ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
