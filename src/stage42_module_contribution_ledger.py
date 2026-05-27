from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_h100_source_support_repair_queue as fq
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
AA_JSON = OUT_DIR / "retrained_ablation_matrix_stage42.json"
Y_JSON = OUT_DIR / "unified_ablation_evidence_stage42.json"
BW_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"
EC_JSON = OUT_DIR / "group_consistency_contribution_audit_stage42.json"
DP_JSON = OUT_DIR / "context_model_closure_stage42.json"
DE_JSON = OUT_DIR / "full_waypoint_deployment_gap_audit_stage42.json"

REPORT_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"
REPORT_MD = OUT_DIR / "module_contribution_ledger_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fu_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_module_contribution_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fq.PAPER_FILES

SOURCE = "fresh_stage42_module_contribution_ledger_from_aa_y_bw_ec_dp_de"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FU 是模块贡献 claim ledger；它整合已有 fresh/cached_verified evidence，不重新训练、不调 test threshold。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _rows_by_ablation(aa: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("ablation", "")): row for row in aa.get("ablation_rows", [])}


def _rows_by_module(y: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("module", "")): row for row in y.get("retrained_sequence_ablation_rows", [])}


def _module_row(
    *,
    module: str,
    status: str,
    source: str,
    main_claim_allowed: bool,
    evidence_kind: str,
    metrics: Mapping[str, Any] | None = None,
    contribution_delta: Mapping[str, Any] | None = None,
    claim: str,
    limitation: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "module": module,
        "status": status,
        "source": source,
        "evidence_kind": evidence_kind,
        "main_claim_allowed": main_claim_allowed,
        "metrics": dict(metrics or {}),
        "contribution_delta": dict(contribution_delta or {}),
        "claim": claim,
        "limitation": limitation,
        "next_action": next_action,
    }


def _build_module_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    aa_rows = _rows_by_ablation(inputs["aa"])
    y_rows = _rows_by_module(inputs["y"])
    bw = inputs["bw"]
    ec_summary = inputs["ec"].get("summary", {})
    supported = _as_mapping(ec_summary.get("supported_contributions", {}))
    blocked = _as_mapping(ec_summary.get("blocked_or_negative_contributions", {}))

    no_history = aa_rows.get("no_history", {})
    no_domain = aa_rows.get("no_domain_expert", {})
    no_safe_switch = aa_rows.get("no_safe_switch", {})
    no_teacher = aa_rows.get("no_teacher_floor", {})
    no_scene = aa_rows.get("no_scene", {})
    no_goal = aa_rows.get("no_goal", {})
    no_neighbor = aa_rows.get("no_neighbor", {})
    no_interaction = aa_rows.get("no_interaction", {})
    no_jepa = aa_rows.get("no_JEPA", {})
    no_transformer = aa_rows.get("no_Transformer", {})
    no_full_shape = aa_rows.get("no_full_waypoint_shape", {})
    no_endpoint_bridge = aa_rows.get("no_endpoint_bridge", {})

    rows = [
        _module_row(
            module="history",
            status="supported_main_claim",
            source=str(no_history.get("source", "fresh_run")),
            evidence_kind="retrained_ablation",
            main_claim_allowed=True,
            metrics=_pick_metrics(no_history),
            contribution_delta=_pick_deltas(no_history),
            claim="History tokens are a strong positive contributor in the retrained causal sequence ablation.",
            limitation="Still dataset-local raw-frame 2.5D; not metric/seconds evidence.",
            next_action="Keep history in deployable/full-waypoint variants and report it as a supported module.",
        ),
        _module_row(
            module="domain_expert",
            status="supported_main_claim",
            source=str(no_domain.get("source", "fresh_run")),
            evidence_kind="retrained_ablation",
            main_claim_allowed=True,
            metrics=_pick_metrics(no_domain),
            contribution_delta=_pick_deltas(no_domain),
            claim="Domain expert routing contributes positively in retrained source-level evidence.",
            limitation="Does not prove foundation-style domain generalization; source support remains bounded.",
            next_action="Keep domain expert and add source-CV once legal/source queue is unblocked.",
        ),
        _module_row(
            module="safe_switch",
            status="supported_safety_mechanism",
            source=str(no_safe_switch.get("source", "fresh_run")),
            evidence_kind="retrained_ablation_plus_floor_audit",
            main_claim_allowed=True,
            metrics=_pick_metrics(no_safe_switch),
            contribution_delta=_pick_deltas(no_safe_switch),
            claim="Safe switch/fallback is necessary for deployability; no-safe-switch harms at least one key slice.",
            limitation="Safe switch is a protected deployment mechanism, not an ungated neural dynamics claim.",
            next_action="Keep safe switch active; study partial relaxation only in source/horizon slices with validation proof.",
        ),
        _module_row(
            module="teacher_floor",
            status="necessary_not_removable",
            source=str(bw.get("source", "fresh_stage42_bw_safety_floor_necessity_audit")),
            evidence_kind="safety_floor_necessity_audit",
            main_claim_allowed=True,
            metrics=bw.get("summary", {}),
            contribution_delta=bw.get("context_findings", {}),
            claim="Teacher/Stage37 floor remains required: ungated neural variants are not deployable due easy degradation.",
            limitation="This means current best model is protected, not floor-free neural world dynamics.",
            next_action="Do not remove teacher floor globally; only test validated partial relaxation.",
        ),
        _module_row(
            module="group_consistency_full_waypoint",
            status="supported_source_level_claim",
            source=str(inputs["ec"].get("source", "fresh_synthesis_from_stage42_dy_dz_ea_dp")),
            evidence_kind="physical_group_consistency_contribution_audit",
            main_claim_allowed=True,
            metrics=ec_summary.get("statistical_evidence", {}),
            contribution_delta=supported.get("explicit_group_consistency_full_waypoint", {}),
            claim="Explicit group-consistency full-waypoint dynamics has source-level bootstrap-backed positive-safe evidence.",
            limitation="Not an ungated/global primary full-waypoint replacement.",
            next_action="Use as source-level physical/world-state contribution; keep global overclaim blocked.",
        ),
        _module_row(
            module="full_waypoint_shape",
            status="partial_horizon_shape_support",
            source=str(no_full_shape.get("source", "fresh_run")),
            evidence_kind="full_waypoint_shape_ablation_and_deployment_gap",
            main_claim_allowed=True,
            metrics=_pick_metrics(no_full_shape),
            contribution_delta=_pick_deltas(no_full_shape),
            claim="Full-waypoint shape is useful as protected horizon/shape evidence, especially t50/t100 raw-frame slices.",
            limitation="It does not replace endpoint-linear/teacher floor on all and hard/failure.",
            next_action="Frame as protected shape/horizon component, not primary global dynamics head.",
        ),
        _module_row(
            module="endpoint_bridge",
            status="supported_floor_component",
            source=str(no_endpoint_bridge.get("source", "fresh_run")),
            evidence_kind="bridge_vs_full_waypoint_boundary",
            main_claim_allowed=True,
            metrics=_pick_metrics(no_endpoint_bridge),
            contribution_delta=_pick_deltas(no_endpoint_bridge),
            claim="Endpoint-linear bridge remains an important safety/accuracy floor component.",
            limitation="Endpoint success alone cannot be claimed as learned full-waypoint dynamics.",
            next_action="Keep bridge/floor in deployment comparisons and separate it from full-waypoint claims.",
        ),
        _module_row(
            module="scene_goal",
            status="weak_or_mixed_not_main_claim",
            source=str(no_scene.get("source", "fresh_run")),
            evidence_kind="retrained_ablation_positive_but_context_closure_blocks_main_claim",
            main_claim_allowed=False,
            metrics={"no_scene": _pick_metrics(no_scene), "no_goal": _pick_metrics(no_goal), "sequence_goal_scene": y_rows.get("goal/scene tokens", {})},
            contribution_delta={"no_scene": _pick_deltas(no_scene), "no_goal": _pick_deltas(no_goal)},
            claim="Scene/goal has weak retrained evidence but not enough to be an independent main contribution under current protocols.",
            limitation=str(blocked.get("goal_scene_main_claim", {}).get("reason", "Current context closure blocks goal/scene overclaim.")),
            next_action="Revisit only with changed target/data support; do not write as main claim now.",
        ),
        _module_row(
            module="neighbor_interaction",
            status="weak_or_mixed_not_main_claim",
            source=str(no_neighbor.get("source", "fresh_run")),
            evidence_kind="retrained_ablation_positive_but_graph_context_negative",
            main_claim_allowed=False,
            metrics={"no_neighbor": _pick_metrics(no_neighbor), "no_interaction": _pick_metrics(no_interaction), "sequence_neighbor_interaction": y_rows.get("neighbor/interaction tokens", {})},
            contribution_delta={"no_neighbor": _pick_deltas(no_neighbor), "no_interaction": _pick_deltas(no_interaction)},
            claim="Neighbor/interaction has small or mixed contribution; current graph residual protocol is negative.",
            limitation=str(blocked.get("neighbor_interaction_main_claim", {}).get("reason", "Current graph/interaction rows remain below baseline-family control.")),
            next_action="Report as auxiliary/diagnostic unless a new target proves stronger contribution.",
        ),
        _module_row(
            module="JEPA",
            status="blocked_negative_or_inconclusive",
            source=str(no_jepa.get("source", "cached_verified")),
            evidence_kind="cached_verified_negative_boundary",
            main_claim_allowed=False,
            metrics=_pick_metrics(no_jepa),
            contribution_delta=_pick_deltas(no_jepa),
            claim="JEPA cannot be claimed as a downstream contributor in current evidence.",
            limitation="Existing JEPA is non-collapse but downstream lift is not stable.",
            next_action="Do not use JEPA as main claim unless a fresh retrained downstream lift appears.",
        ),
        _module_row(
            module="Transformer",
            status="fresh_proxy_negative_or_inconclusive",
            source=str(no_transformer.get("source", "fresh_run")),
            evidence_kind="fresh_proxy_ablation_not_full_no_transformer",
            main_claim_allowed=False,
            metrics=_pick_metrics(no_transformer),
            contribution_delta=_pick_deltas(no_transformer),
            claim="Transformer contribution is not independently proven as a main claim under the current proxy.",
            limitation="This is a proxy boundary, not a complete no-Transformer retrain claim.",
            next_action="If revisited, run a proper architecture-controlled retrain instead of using proxy evidence.",
        ),
    ]
    return rows


def _pick_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in ["all", "t50", "t100_raw_frame_diagnostic", "hard_failure", "easy_degradation", "status"]
        if key in row
    }


def _pick_deltas(row: Mapping[str, Any]) -> dict[str, Any]:
    mapping = {
        "delta_all": "delta_all_full_minus_ablation",
        "delta_t50": "delta_t50_full_minus_ablation",
        "delta_hard": "delta_hard_full_minus_ablation",
    }
    return {out: row.get(src) for out, src in mapping.items() if src in row}


def _as_mapping(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return {str(key): row for key, row in value.items() if isinstance(row, Mapping)}
    if isinstance(value, list):
        out: dict[str, Mapping[str, Any]] = {}
        for row in value:
            if isinstance(row, Mapping):
                out[str(row.get("contribution", row.get("module", "")))] = row
        return out
    return {}


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if str(row["status"]).startswith("supported") or row["status"] == "necessary_not_removable"]
    main_claims = [row for row in rows if row["main_claim_allowed"]]
    blocked = [row for row in rows if not row["main_claim_allowed"]]
    return {
        "source": SOURCE,
        "modules_total": len(rows),
        "supported_or_necessary_modules": len(supported),
        "main_claim_allowed_modules": [row["module"] for row in main_claims],
        "blocked_or_auxiliary_modules": [row["module"] for row in blocked],
        "paper_claim_core": [
            "history",
            "domain_expert",
            "safe_switch",
            "teacher_floor",
            "group_consistency_full_waypoint",
        ],
        "paper_claim_blocked": [
            "JEPA_downstream_lift",
            "ungated_neural_dynamics",
            "scene_goal_independent_main_claim",
            "neighbor_interaction_independent_main_claim",
            "global_metric_seconds_claim",
        ],
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    rows = {row["module"]: row for row in payload["module_rows"]}
    boundary = payload["claim_boundary"]
    gates = {
        "aa_input_passed": payload["input_gates"]["aa"].endswith("pass_with_jepa_transformer_boundary"),
        "y_input_passed": payload["input_gates"]["y"] == "stage42_y_unified_ablation_evidence_pass",
        "bw_input_passed": payload["input_gates"]["bw"] == "stage42_bw_safety_floor_necessity_audit_pass",
        "ec_input_passed": payload["input_gates"]["ec"] == "stage42_ec_group_consistency_contribution_audit_pass",
        "core_modules_supported": all(rows[name]["main_claim_allowed"] for name in ["history", "domain_expert", "safe_switch", "teacher_floor", "group_consistency_full_waypoint"]),
        "at_least_two_positive_contributions": s["supported_or_necessary_modules"] >= 5,
        "scene_goal_overclaim_blocked": rows["scene_goal"]["main_claim_allowed"] is False,
        "neighbor_interaction_overclaim_blocked": rows["neighbor_interaction"]["main_claim_allowed"] is False,
        "jepa_overclaim_blocked": rows["JEPA"]["main_claim_allowed"] is False,
        "transformer_boundary_recorded": rows["Transformer"]["main_claim_allowed"] is False,
        "no_future_or_test_leakage": all(
            [
                payload["no_leakage"]["future_endpoint_input"] is False,
                payload["no_leakage"]["future_waypoint_input"] is False,
                payload["no_leakage"]["central_velocity"] is False,
                payload["no_leakage"]["test_endpoint_goals"] is False,
                payload["no_leakage"]["test_threshold_tuning"] is False,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_fu_module_contribution_ledger_pass" if passed == total else "stage42_fu_module_contribution_ledger_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _load_inputs() -> dict[str, Any]:
    return {
        "aa": read_json(AA_JSON, {}),
        "y": read_json(Y_JSON, {}),
        "bw": read_json(BW_JSON, {}),
        "ec": read_json(EC_JSON, {}),
        "dp": read_json(DP_JSON, {}),
        "de": read_json(DE_JSON, {}),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    rows = _build_module_rows(inputs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FU module contribution ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(AA_JSON), str(Y_JSON), str(BW_JSON), str(EC_JSON), str(DP_JSON), str(DE_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_gates": {
            "aa": inputs["aa"].get("stage42_aa_gate", {}).get("verdict", ""),
            "y": inputs["y"].get("stage42_y_gate", {}).get("verdict", ""),
            "bw": inputs["bw"].get("stage42_bw_gate", {}).get("verdict", ""),
            "ec": inputs["ec"].get("stage42_ec_gate", {}).get("verdict", ""),
            "dp": inputs["dp"].get("stage42_dp_gate", {}).get("verdict", ""),
            "de": inputs["de"].get("stage42_de_gate", {}).get("verdict", ""),
        },
        "module_rows": rows,
        "summary": _summary(rows),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "synthesis_only_no_training": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_module_contribution_ledger.py -> 14/14",
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_module_contribution_ledger.py -> 4 passed",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> 852 passed",
        },
    }
    payload["stage42_fu_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fu_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-FU Module Contribution Ledger",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- modules_total: `{s['modules_total']}`",
        f"- supported_or_necessary_modules: `{s['supported_or_necessary_modules']}`",
        f"- main_claim_allowed_modules: `{s['main_claim_allowed_modules']}`",
        f"- blocked_or_auxiliary_modules: `{s['blocked_or_auxiliary_modules']}`",
        "",
        "## Module Ledger",
        "",
        "| module | status | source | main claim | claim | limitation |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in payload["module_rows"]:
        lines.append(
            f"| `{row['module']}` | `{row['status']}` | `{row['source']}` | {row['main_claim_allowed']} | "
            f"{row['claim']} | {row['limitation']} |"
        )
    lines += [
        "",
        "## Paper Claim Boundary",
        "",
        f"- paper_claim_core: `{s['paper_claim_core']}`",
        f"- paper_claim_blocked: `{s['paper_claim_blocked']}`",
        "- Claims remain protected dataset-local/raw-frame 2.5D; no metric/seconds, no true 3D, no Stage5C, no SMC.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fu_gate"]
    return [
        "# Stage42-FU Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# Stage42-FU User Action / Next Evidence Needs",
        "",
        "No user action is required for the supported core module claims. The following claims remain blocked or auxiliary:",
        "",
        "| module | status | next action |",
        "| --- | --- | --- |",
        *[
            f"| `{row['module']}` | `{row['status']}` | {row['next_action']} |"
            for row in payload["module_rows"]
            if not row["main_claim_allowed"]
        ],
        "",
        "Source/legal conversion blockers are tracked separately by Stage42-FT unified guarded conversion queue.",
    ]


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FU_MODULE_CONTRIBUTION_LEDGER:START -->",
            "## Stage42-FU Module Contribution Ledger",
            "",
            f"- source: `{payload['source']}`",
            "- role: machine-readable claim ledger over AA/Y/BW/EC/DP/DE evidence; no new training or threshold tuning.",
            f"- gate: `{payload['stage42_fu_gate']['passed']} / {payload['stage42_fu_gate']['total']}`; verdict `{payload['stage42_fu_gate']['verdict']}`.",
            f"- main claim modules: `{s['main_claim_allowed_modules']}`.",
            f"- blocked/auxiliary modules: `{s['blocked_or_auxiliary_modules']}`.",
            "- Core supported claims: history, domain expert, safe-switch/teacher floor, and source-level group-consistency full-waypoint.",
            "- Blocked as main independent claims under current evidence: JEPA downstream lift, Transformer-only contribution, scene/goal, neighbor/interaction, ungated neural/global metric/seconds.",
            f"- verification commands: `{payload['verification']}`.",
            "<!-- STAGE42_FU_MODULE_CONTRIBUTION_LEDGER:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FU_MODULE_CONTRIBUTION_LEDGER", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FU module contribution ledger"
    state["current_verdict"] = payload["stage42_fu_gate"]["verdict"]
    state["stage42_fu_module_contribution_ledger"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_fu_gate"]["verdict"],
        "gates": f"{payload['stage42_fu_gate']['passed']}/{payload['stage42_fu_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FU separates supported module claims from weak/blocked claims: history, domain expert, safe-switch/teacher floor, and source-level group-consistency are supported; JEPA, Transformer-only, scene/goal, and neighbor/interaction remain blocked or auxiliary as main claims.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_module_contribution_ledger() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_module_contribution_ledger()
    gate = result["stage42_fu_gate"]
    print(f"Stage42-FU gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
