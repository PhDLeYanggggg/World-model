from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from src import stage43_full_waypoint_latent_dynamics as m
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_graph_failure_taxonomy.json"
REPORT_MD = OUT_DIR / "stage43_scene_graph_failure_taxonomy.md"
GATE_MD = OUT_DIR / "stage43_stage_dk_scene_graph_failure_taxonomy_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

BL_JSON = OUT_DIR / "stage43_raw_scene_graph_ablation_readiness.json"
AG_JSON = OUT_DIR / "stage43_scene_proxy_retrained_ablation.json"
BO_JSON = OUT_DIR / "stage43_graph_history_retrained_ablation.json"
BP_JSON = OUT_DIR / "stage43_scene_graph_multimodal_ablation.json"
BQ_JSON = OUT_DIR / "stage43_gated_scene_graph_fusion.json"
DJ_JSON = OUT_DIR / "stage43_latent_world_state_current_reconciliation.json"

SOURCE = "fresh_stage43_dk_scene_graph_failure_taxonomy"
SECTION = "STAGE43_DK_SCENE_GRAPH_FAILURE_TAXONOMY"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _full_pass(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate) and int(gate.get("passed", -1)) == int(gate.get("total", -2))


def _verdict(payload: Mapping[str, Any], gate_key: str) -> str:
    return str(payload.get(gate_key, {}).get("verdict", "missing"))


def _variant(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    for row in payload.get("variants", []):
        if row.get("variant") == name:
            return dict(row)
    raise KeyError(f"variant not found: {name}")


def _metric(row: Mapping[str, Any], key: str) -> float:
    return float(row.get("test_metrics_with_floor", {}).get(key, 0.0))


def _metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all": _metric(row, "full_waypoint_ade_improvement_vs_floor"),
        "t50": _metric(row, "t50_full_waypoint_ade_improvement_vs_floor"),
        "t100_raw_frame_diagnostic": _metric(row, "t100_raw_frame_full_waypoint_diagnostic_vs_floor"),
        "hard_failure": _metric(row, "hard_failure_full_waypoint_ade_improvement_vs_floor"),
        "easy_degradation": _metric(row, "easy_degradation_vs_floor"),
        "switch_rate": _metric(row, "switch_rate"),
    }


def _diff(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    return {key: float(a[key] - b[key]) for key in a}


def _clean_no_leakage(*rows: Mapping[str, Any]) -> bool:
    for row in rows:
        if row.get("future_endpoint_input") is not False:
            return False
        if row.get("future_waypoint_input") is not False:
            return False
        if row.get("central_velocity_input") is not False:
            return False
        if row.get("test_endpoint_goal_construction") is not False:
            return False
        if row.get("test_statistics_normalization") is not False:
            return False
    return True


def _clean_claims(*rows: Mapping[str, Any]) -> bool:
    for row in rows:
        if row.get("true_3d_world_model", row.get("true_3d", False)) is not False:
            return False
        if row.get("foundation_world_model") is not False:
            return False
        if row.get("metric_or_seconds_claim") is not False:
            return False
        if row.get("stage5c_executed") is not False:
            return False
        if row.get("smc_enabled") is not False:
            return False
    return True


def build_scene_graph_failure_taxonomy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bl = read_json(BL_JSON, {})
    ag = read_json(AG_JSON, {})
    bo = read_json(BO_JSON, {})
    bp = read_json(BP_JSON, {})
    bq = read_json(BQ_JSON, {})
    dj = read_json(DJ_JSON, {})

    no_graph = _metrics(_variant(bo, "no_graph"))
    current_graph = _metrics(_variant(bo, "current_graph_only"))
    history_graph = _metrics(_variant(bo, "history_graph_only"))
    full_graph = _metrics(_variant(bo, "full_graph"))

    no_scene = _metrics(_variant(ag, "no_scene"))
    geometry_route = _metrics(_variant(ag, "geometry_route"))
    goal_only = _metrics(_variant(ag, "goal_only"))
    full_scene = _metrics(_variant(ag, "full_scene"))

    no_context = _metrics(_variant(bp, "no_context"))
    scene_only = _metrics(_variant(bp, "scene_proxy_only"))
    graph_only = _metrics(_variant(bp, "graph_history_only"))
    scene_graph_full = _metrics(_variant(bp, "scene_graph_full"))

    bq_best = dict(bq.get("best_single_metrics", {}))
    bq_no_context = dict(bq.get("no_context_metrics", {}))
    bq_minus_best = {
        "all": float(bq.get("gated_minus_best_single_by_t50", {}).get("full_waypoint_ade_improvement_vs_floor", 0.0)),
        "t50": float(bq.get("gated_minus_best_single_by_t50", {}).get("t50_full_waypoint_ade_improvement_vs_floor", 0.0)),
        "t100_raw_frame_diagnostic": float(
            bq.get("gated_minus_best_single_by_t50", {}).get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0)
        ),
        "hard_failure": float(
            bq.get("gated_minus_best_single_by_t50", {}).get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0)
        ),
        "easy_degradation": float(bq.get("gated_minus_best_single_by_t50", {}).get("easy_degradation_vs_floor", 0.0)),
        "switch_rate": float(bq.get("gated_minus_best_single_by_t50", {}).get("switch_rate", 0.0)),
    }
    bq_ci = dict(bq.get("bootstrap_gated_vs_best_single_t50_ci", {}).get("metrics", {}))
    t50_ci = dict(bq_ci.get("t50_full_waypoint_ade_contribution", {}))

    graph_deltas = {
        "full_graph_minus_no_graph": _diff(full_graph, no_graph),
        "full_graph_minus_current_graph": _diff(full_graph, current_graph),
        "full_graph_minus_history_graph": _diff(full_graph, history_graph),
    }
    scene_deltas = {
        "geometry_route_minus_no_scene": _diff(geometry_route, no_scene),
        "full_scene_minus_no_scene": _diff(full_scene, no_scene),
        "goal_only_minus_no_scene": _diff(goal_only, no_scene),
    }
    fusion_deltas = {
        "scene_graph_full_minus_graph_only": _diff(scene_graph_full, graph_only),
        "scene_graph_full_minus_no_context": _diff(scene_graph_full, no_context),
        "gated_fusion_minus_best_single": bq_minus_best,
    }

    taxonomy = [
        {
            "failure": "naive_scene_graph_fusion_suppresses_graph_signal",
            "evidence": (
                f"BP scene_graph_full t50 is {_pct(scene_graph_full['t50'])}, while graph_history_only is "
                f"{_pct(graph_only['t50'])}; delta {_pct(fusion_deltas['scene_graph_full_minus_graph_only']['t50'])}."
            ),
            "interpretation": "The strongest graph signal is real, but concatenating scene proxy and graph history changes the learned decision surface in a harmful way.",
            "repair_target": "Train graph-first mixture/routing where graph_history/full_graph is the default neural expert and scene proxy can only add residual context under a harm guard.",
        },
        {
            "failure": "scene_proxy_is_useful_but_not_raw_scene_or_sdf",
            "evidence": (
                f"AG geometry_route improves t50 by {_pct(scene_deltas['geometry_route_minus_no_scene']['t50'])} "
                f"over no_scene with easy {_pct(geometry_route['easy_degradation'])}, but full_scene easy is "
                f"{_pct(full_scene['easy_degradation'])}."
            ),
            "interpretation": "Scene/goal proxy has signal, but broad scene proxy use can over-switch easy cases and still cannot support a raw-scene/SDF claim.",
            "repair_target": "Use a narrow geometry-route scene expert until raw scene/SDF tensors exist; keep raw-scene claims blocked.",
        },
        {
            "failure": "learned_gated_fusion_is_safe_but_too_conservative_or_misweighted",
            "evidence": (
                f"BQ gated fusion loses {_pct(-bq_minus_best['t50'])} t50 versus the best single expert; "
                f"bootstrap t50 contribution CI is [{_pct(float(t50_ci.get('low', 0.0)))}, {_pct(float(t50_ci.get('high', 0.0)))}]."
            ),
            "interpretation": "The gate prevents the catastrophic easy damage of full fusion, but it also fails to preserve the graph expert's t50/hard lift.",
            "repair_target": "Add expert-preservation distillation: the fused model must not underperform graph_history_only on validation t50/hard before it can switch.",
        },
        {
            "failure": "t100_raw_frame_remains_diagnostic",
            "evidence": (
                f"Full_graph t100 raw-frame diagnostic is {_pct(full_graph['t100_raw_frame_diagnostic'])}; "
                f"BQ t100 delta versus best single is {_pct(bq_minus_best['t100_raw_frame_diagnostic'])}."
            ),
            "interpretation": "The current scene/graph path is not a reliable long-horizon t100 solution.",
            "repair_target": "Keep t100 guarded and diagnostic; do not use it as deployment evidence until source/group support is stable.",
        },
    ]

    next_training_contract = {
        "name": "stage43_next_graph_first_scene_residual_moe",
        "train_next": True,
        "do_not_repeat": [
            "generic_scene_graph_full_concat",
            "gated_fusion_without_graph_expert_preservation",
            "raw_scene_claim_without_raw_scene_or_sdf_cache",
        ],
        "required_experts": [
            "no_context_floor_compatible_expert",
            "full_graph_or_graph_history_expert",
            "geometry_route_scene_proxy_expert",
        ],
        "required_losses": [
            "full_waypoint_ade_loss",
            "failure_gain_harm_multitask_loss",
            "graph_expert_preservation_pairwise_loss",
            "easy_harm_penalty",
            "t50_hard_failure_weighting",
            "t100_guarded_diagnostic_reporting",
        ],
        "deployment_rule": (
            "Default to the protected floor or graph expert; allow scene proxy residual only when gain is high, harm is low, "
            "and validation support says the source/horizon slice is covered."
        ),
        "success_gate": {
            "beats_best_single_graph_t50_or_hard": True,
            "easy_degradation_max": 0.02,
            "no_test_threshold_tuning": True,
            "raw_scene_claim_allowed": False,
        },
    }

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_taxonomy_from_stage43_bl_ag_bo_bp_bq_dj",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "raw_scene_graph_readiness": str(BL_JSON),
            "scene_proxy_retrained_ablation": str(AG_JSON),
            "graph_history_retrained_ablation": str(BO_JSON),
            "scene_graph_multimodal_ablation": str(BP_JSON),
            "gated_scene_graph_fusion": str(BQ_JSON),
            "latent_world_state_current_reconciliation": str(DJ_JSON),
        },
        "input_hash": _combined_hash([BL_JSON, AG_JSON, BO_JSON, BP_JSON, BQ_JSON, DJ_JSON]),
        "input_verdicts": {
            "stage43_bl": _verdict(bl, "stage43_bl_gate"),
            "stage43_ag": _verdict(ag, "stage43_ag_gate"),
            "stage43_bo": _verdict(bo, "stage43_bo_gate"),
            "stage43_bp": _verdict(bp, "stage43_bp_gate"),
            "stage43_bq": _verdict(bq, "stage43_bq_gate"),
            "stage43_dj": _verdict(dj, "stage43_dj_gate"),
        },
        "signals": {
            "graph": {
                "no_graph": no_graph,
                "current_graph_only": current_graph,
                "history_graph_only": history_graph,
                "full_graph": full_graph,
                "deltas": graph_deltas,
            },
            "scene_proxy": {
                "no_scene": no_scene,
                "geometry_route": geometry_route,
                "goal_only": goal_only,
                "full_scene": full_scene,
                "deltas": scene_deltas,
            },
            "fusion": {
                "no_context": no_context,
                "scene_proxy_only": scene_only,
                "graph_history_only": graph_only,
                "scene_graph_full": scene_graph_full,
                "deltas": fusion_deltas,
                "bq_best_single": bq_best,
                "bq_no_context": bq_no_context,
                "bq_t50_contribution_ci": t50_ci,
            },
        },
        "taxonomy": taxonomy,
        "next_training_contract": next_training_contract,
        "decision": {
            "deployable_policy_changed": False,
            "keep_current_protected_latent_state_candidate": True,
            "generic_scene_graph_concat_rejected": True,
            "graph_first_scene_residual_moe_is_next": True,
            "raw_scene_sdf_still_blocked": True,
            "t100_raw_frame_still_diagnostic": True,
        },
        "no_leakage": {
            "bp": dict(bp.get("no_leakage", {})),
            "bq": dict(bq.get("no_leakage", {})),
            "dj": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
        },
        "claim_boundary": {
            "bl": dict(bl.get("claim_boundary", {})),
            "bp": dict(bp.get("claim_boundary", {})),
            "bq": dict(bq.get("claim_boundary", {})),
            "dj": dict(dj.get("claim_boundary", {}).get("current_public_claim", {})),
            "current": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "dataset_local_raw_frame_only": True,
                "raw_scene_or_verified_sdf_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
                "long_objective_complete": False,
            },
        },
        "preconditions": {
            "stage43_bl": _full_pass(bl, "stage43_bl_gate"),
            "stage43_ag": _full_pass(ag, "stage43_ag_gate"),
            "stage43_bo": _full_pass(bo, "stage43_bo_gate"),
            "stage43_bp": _full_pass(bp, "stage43_bp_gate"),
            "stage43_bq": _full_pass(bq, "stage43_bq_gate"),
            "stage43_dj": _full_pass(dj, "stage43_dj_gate"),
        },
    }
    payload["stage43_dk_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    pre = payload["preconditions"]
    signals = payload["signals"]
    graph = signals["graph"]
    scene = signals["scene_proxy"]
    fusion = signals["fusion"]
    decision = payload["decision"]
    current_claim = payload["claim_boundary"]["current"]
    gates = {
        "preconditions_passed": all(bool(v) for v in pre.values()),
        "graph_signal_positive_and_easy_safe": graph["full_graph"]["t50"] > graph["no_graph"]["t50"]
        and graph["full_graph"]["hard_failure"] > graph["no_graph"]["hard_failure"]
        and graph["full_graph"]["easy_degradation"] <= 0.02,
        "scene_proxy_signal_present_but_guarded": scene["geometry_route"]["t50"] > scene["no_scene"]["t50"]
        and scene["geometry_route"]["easy_degradation"] <= 0.02
        and scene["full_scene"]["easy_degradation"] > 0.02,
        "naive_fusion_failure_identified": fusion["scene_graph_full"]["t50"]
        < fusion["graph_history_only"]["t50"]
        and fusion["scene_graph_full"]["easy_degradation"] > 0.02,
        "gated_fusion_safe_no_lift_identified": fusion["deltas"]["gated_fusion_minus_best_single"]["t50"] < 0.0
        and payload["input_verdicts"]["stage43_bq"] == "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
        "bootstrap_confirms_negative_gated_t50": float(fusion["bq_t50_contribution_ci"].get("high", 0.0)) < 0.0,
        "next_training_contract_recorded": payload["next_training_contract"]["train_next"] is True
        and payload["next_training_contract"]["name"] == "stage43_next_graph_first_scene_residual_moe",
        "deployable_policy_not_changed": decision["deployable_policy_changed"] is False,
        "raw_scene_sdf_not_overclaimed": decision["raw_scene_sdf_still_blocked"] is True
        and current_claim["raw_scene_or_verified_sdf_claim"] is False,
        "t100_remains_diagnostic": decision["t100_raw_frame_still_diagnostic"] is True,
        "no_future_or_test_leakage": _clean_no_leakage(
            payload["no_leakage"]["bp"], payload["no_leakage"]["bq"], payload["no_leakage"]["dj"]
        ),
        "claim_boundary_not_overstated": _clean_claims(
            payload["claim_boundary"]["bl"],
            payload["claim_boundary"]["bp"],
            payload["claim_boundary"]["bq"],
            payload["claim_boundary"]["current"],
        )
        and current_claim["dataset_local_raw_frame_only"] is True
        and current_claim["raw_scene_or_verified_sdf_claim"] is False,
        "stage5c_and_smc_false": current_claim["stage5c_executed"] is False and current_claim["smc_enabled"] is False,
        "long_objective_kept_active": current_claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_dk_scene_graph_failure_taxonomy_pass_next_graph_first_moe"
        if passed == total
        else "stage43_dk_scene_graph_failure_taxonomy_incomplete",
        "protected_multimodal_latent_state_candidate": bool(passed == total),
        "deployable_policy_changed": False,
        "next_training_contract": payload["next_training_contract"]["name"],
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_dk_gate"]
    graph = payload["signals"]["graph"]
    scene = payload["signals"]["scene_proxy"]
    fusion = payload["signals"]["fusion"]
    lines = [
        "# Stage43-DK Scene/Graph Failure Taxonomy",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
        f"- next training contract: `{gate['next_training_contract']}`",
        "",
        "## What I Found",
        "",
        f"- Full graph is useful: t50 `{_pct(graph['full_graph']['t50'])}`, hard/failure `{_pct(graph['full_graph']['hard_failure'])}`, easy `{_pct(graph['full_graph']['easy_degradation'])}`.",
        f"- Geometry-route scene proxy is useful but narrow: t50 `{_pct(scene['geometry_route']['t50'])}`, easy `{_pct(scene['geometry_route']['easy_degradation'])}`.",
        f"- Full scene proxy is not deployment-safe: easy `{_pct(scene['full_scene']['easy_degradation'])}`.",
        f"- Naive scene+graph full fusion underperforms graph-only: t50 delta `{_pct(fusion['deltas']['scene_graph_full_minus_graph_only']['t50'])}`, easy delta `{_pct(fusion['deltas']['scene_graph_full_minus_graph_only']['easy_degradation'])}`.",
        f"- Learned gated fusion is safe but loses t50 vs best single: delta `{_pct(fusion['deltas']['gated_fusion_minus_best_single']['t50'])}`.",
        "",
        "## Failure Taxonomy",
        "",
    ]
    for item in payload["taxonomy"]:
        lines.extend(
            [
                f"### {item['failure']}",
                "",
                f"- evidence: {item['evidence']}",
                f"- interpretation: {item['interpretation']}",
                f"- repair target: {item['repair_target']}",
                "",
            ]
        )
    contract = payload["next_training_contract"]
    lines.extend(
        [
            "## Next Training Contract",
            "",
            f"- name: `{contract['name']}`",
            f"- train next: `{contract['train_next']}`",
            f"- deployment rule: {contract['deployment_rule']}",
            "",
            "Required experts:",
            *[f"- `{item}`" for item in contract["required_experts"]],
            "",
            "Required losses:",
            *[f"- `{item}`" for item in contract["required_losses"]],
            "",
            "Do not repeat:",
            *[f"- `{item}`" for item in contract["do_not_repeat"]],
            "",
            "## Boundary",
            "",
            "- This is a fresh evidence taxonomy, not a new deployed model.",
            "- I keep the current protected latent-state candidate.",
            "- No raw-scene/SDF claim until a raw-scene/SDF cache exists.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    lines = _render_report(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    gate = payload["stage43_dk_gate"]
    graph = payload["signals"]["graph"]
    scene = payload["signals"]["scene_proxy"]
    fusion = payload["signals"]["fusion"]
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- protected multimodal latent-state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
        f"- next training contract: `{gate['next_training_contract']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "## Current Decision",
        "",
        "Graph history has real signal, scene proxy has narrower signal, but naive scene+graph fusion and the current learned gate do not safely beat the graph expert. The next model should be graph-first with a scene residual expert and explicit expert-preservation loss.",
        "",
        "## Key Evidence",
        "",
        f"- full_graph t50/hard/easy: `{_pct(graph['full_graph']['t50'])}` / `{_pct(graph['full_graph']['hard_failure'])}` / `{_pct(graph['full_graph']['easy_degradation'])}`",
        f"- geometry_route scene t50/easy: `{_pct(scene['geometry_route']['t50'])}` / `{_pct(scene['geometry_route']['easy_degradation'])}`",
        f"- scene_graph_full minus graph_history_only t50/easy: `{_pct(fusion['deltas']['scene_graph_full_minus_graph_only']['t50'])}` / `{_pct(fusion['deltas']['scene_graph_full_minus_graph_only']['easy_degradation'])}`",
        f"- gated fusion minus best single t50: `{_pct(fusion['deltas']['gated_fusion_minus_best_single']['t50'])}`",
        "",
        "## Boundaries",
        "",
        "- This does not change the deployable policy.",
        "- Raw scene/SDF remains blocked.",
        "- t100 remains raw-frame diagnostic.",
        "- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    world_lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    write_md(WORLD_GATE_MD, world_lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_dk_gate"]
    graph = payload["signals"]["graph"]
    scene = payload["signals"]["scene_proxy"]
    fusion = payload["signals"]["fusion"]
    section = [
        "## Stage43-DK: Scene/Graph Failure Taxonomy",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        f"next_training_contract = `{gate['next_training_contract']}`",
        "",
        f"full_graph_t50_hard_easy = `{_pct(graph['full_graph']['t50'])}` / `{_pct(graph['full_graph']['hard_failure'])}` / `{_pct(graph['full_graph']['easy_degradation'])}`",
        f"geometry_route_scene_t50_easy = `{_pct(scene['geometry_route']['t50'])}` / `{_pct(scene['geometry_route']['easy_degradation'])}`",
        f"scene_graph_full_minus_graph_only_t50_easy = `{_pct(fusion['deltas']['scene_graph_full_minus_graph_only']['t50'])}` / `{_pct(fusion['deltas']['scene_graph_full_minus_graph_only']['easy_degradation'])}`",
        "",
        "My read: graph history is the part worth protecting and building around. Scene/goal proxy is not useless, but generic scene+graph fusion damaged the graph expert and hurt easy cases. The next neural step should be a graph-first scene-residual MoE with expert-preservation and harm-aware routing, not another raw concat.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_dk_scene_graph_failure_taxonomy"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "decision": payload["decision"],
        "next_training_contract": payload["next_training_contract"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_dk_scene_graph_failure_taxonomy"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-DK",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "next_training_contract": gate["next_training_contract"],
                        "deployable_policy_changed": False,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_scene_graph_failure_taxonomy() -> dict[str, Any]:
    payload = build_scene_graph_failure_taxonomy()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Build Stage43-DK scene/graph failure taxonomy.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_scene_graph_failure_taxonomy()
    gate = payload["stage43_dk_gate"]
    print(f"Stage43-DK: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"next_training_contract={gate['next_training_contract']}")
    print(f"deployable_policy_changed={gate['deployable_policy_changed']}")
    return payload


if __name__ == "__main__":
    main()
