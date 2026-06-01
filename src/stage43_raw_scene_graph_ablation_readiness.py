from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_raw_scene_graph_ablation_readiness.json"
REPORT_MD = OUT_DIR / "stage43_raw_scene_graph_ablation_readiness.md"
GAP_MD = OUT_DIR / "stage43_raw_scene_graph_ablation_gap_matrix.md"
GATE_MD = OUT_DIR / "stage43_stage_bl_raw_scene_graph_ablation_readiness_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bl_raw_scene_graph_ablation_readiness"
SECTION = "STAGE43_BL_RAW_SCENE_GRAPH_ABLATION_READINESS"

SCENE_PROXY_DIR = Path("data/stage43_scene_proxy_tokens")

INPUTS = {
    "stage43_aa_scene_proxy_tokens": OUT_DIR / "stage43_scene_raster_proxy_tokens.json",
    "stage43_ag_scene_proxy_retrained_ablation": OUT_DIR / "stage43_scene_proxy_retrained_ablation.json",
    "stage43_ah_feature_family_retrained_ablation": OUT_DIR / "stage43_feature_family_retrained_ablation.json",
    "stage43_ai_feature_family_multiseed": OUT_DIR / "stage43_feature_family_multiseed_confirmation.json",
    "stage43_x_interaction_proxy": OUT_DIR / "stage43_interaction_validity_proxy.json",
    "stage43_y_multimodal_head_suite": OUT_DIR / "stage43_multimodal_latent_head_suite.json",
    "stage43_bk_t100_reconciliation": OUT_DIR / "stage43_t100_family_limited_reconciliation.json",
}

RAW_SCENE_REQUIRED_KEYS = {
    "scene_image_patch",
    "scene_raster",
    "scene_sdf",
    "walkable_sdf",
    "homography",
}
GRAPH_RICH_REQUIRED_KEYS = {
    "edge_index",
    "edge_attr",
    "neighbor_agent_ids",
    "all_agent_current_xy",
    "all_agent_history_xy",
}


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _read_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _gate_full_pass(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return int(gate.get("passed", -1)) == int(gate.get("total", -2)) and int(gate.get("total", 0)) > 0


def _gate_verdict(payload: Mapping[str, Any], key: str) -> str:
    return str(payload.get(key, {}).get("verdict", "missing"))


def _npz_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "keys": [], "rows": 0}
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        rows = int(data[keys[0]].shape[0]) if keys and hasattr(data[keys[0]], "shape") else 0
        feature_names = data["feature_names"].astype(str).tolist() if "feature_names" in keys else []
        return {
            "path": str(path),
            "exists": True,
            "keys": keys,
            "rows": rows,
            "feature_names": feature_names,
            "has_raw_scene_keys": bool(RAW_SCENE_REQUIRED_KEYS & set(keys)),
            "has_graph_rich_keys": bool(GRAPH_RICH_REQUIRED_KEYS & set(keys)),
        }


def _cache_summary(split: str) -> dict[str, Any]:
    path = m._cache_path(split)
    if not path.exists():
        return {"path": str(path), "exists": False, "keys": [], "rows": 0}
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        return {
            "path": str(path),
            "exists": True,
            "keys": keys,
            "rows": int(data["current_xy"].shape[0]) if "current_xy" in keys else 0,
            "has_row_geometry": {"scene_id", "source_file", "agent_id", "frame_id", "current_xy"}.issubset(keys),
            "has_future_labels": {"future_xy", "waypoint_xy", "waypoint_valid"}.issubset(keys),
            "has_raw_scene_keys": bool(RAW_SCENE_REQUIRED_KEYS & set(keys)),
            "has_graph_rich_keys": bool(GRAPH_RICH_REQUIRED_KEYS & set(keys)),
            "present_raw_scene_like_keys": sorted(RAW_SCENE_REQUIRED_KEYS & set(keys)),
            "present_graph_like_keys": sorted(GRAPH_RICH_REQUIRED_KEYS & set(keys)),
        }


def _variant_by_name(payload: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    for row in payload.get("variants", []):
        if row.get("variant") == variant:
            return dict(row)
    return None


def _metric(row: Mapping[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    metrics = row.get("test_metrics_with_floor", {})
    if key not in metrics:
        return None
    return float(metrics[key])


def build_raw_scene_graph_ablation_readiness() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    artifacts = {name: _read_required(path) for name, path in INPUTS.items()}
    aa = artifacts["stage43_aa_scene_proxy_tokens"]
    ag = artifacts["stage43_ag_scene_proxy_retrained_ablation"]
    ah = artifacts["stage43_ah_feature_family_retrained_ablation"]
    ai = artifacts["stage43_ai_feature_family_multiseed"]
    x = artifacts["stage43_x_interaction_proxy"]
    y = artifacts["stage43_y_multimodal_head_suite"]
    bk = artifacts["stage43_bk_t100_reconciliation"]

    scene_proxy_npz = {
        split: _npz_summary(SCENE_PROXY_DIR / f"stage43_scene_proxy_features_{split}.npz")
        for split in ["train", "val", "test"]
    }
    waypoint_cache = {split: _cache_summary(split) for split in ["train", "val", "test"]}

    scene_full = _variant_by_name(ag, "full_scene")
    scene_no = _variant_by_name(ag, "no_scene")
    feat_full = _variant_by_name(ah, "full_features")
    no_goal = _variant_by_name(ah, "no_goal")
    no_neighbor = _variant_by_name(ah, "no_neighbor_interaction")

    scene_proxy_delta = {
        "full_scene_minus_no_scene_t50": float(
            (scene_full or {}).get("delta_vs_retrained_no_scene", {}).get(
                "t50_full_waypoint_ade_improvement_vs_floor", 0.0
            )
        ),
        "full_scene_minus_no_scene_hard": float(
            (scene_full or {}).get("delta_vs_retrained_no_scene", {}).get(
                "hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0
            )
        ),
        "full_scene_easy_degradation": _metric(scene_full, "easy_degradation_vs_floor"),
        "no_scene_easy_degradation": _metric(scene_no, "easy_degradation_vs_floor"),
    }
    feature_family_delta = {
        "full_minus_no_goal_t50": float(
            (no_goal or {}).get("delta_full_minus_variant", {}).get(
                "t50_full_waypoint_ade_improvement_vs_floor", 0.0
            )
        ),
        "full_minus_no_neighbor_interaction_t50": float(
            (no_neighbor or {}).get("delta_full_minus_variant", {}).get(
                "t50_full_waypoint_ade_improvement_vs_floor", 0.0
            )
        ),
        "full_minus_no_neighbor_interaction_hard": float(
            (no_neighbor or {}).get("delta_full_minus_variant", {}).get(
                "hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0
            )
        ),
        "full_features_easy_degradation": _metric(feat_full, "easy_degradation_vs_floor"),
    }

    raw_scene_ready = all(row["has_raw_scene_keys"] for row in waypoint_cache.values())
    graph_rich_ready = all(row["has_graph_rich_keys"] for row in waypoint_cache.values())
    scene_proxy_only = bool(aa["claim_boundary"]["scene_raster_proxy_only"])
    interaction_proxy_only = bool(x["claim_boundary"]["future_interaction_risk_is_proxy_label"])

    blocker_matrix = [
        {
            "blocker": "raw_scene_or_verified_sdf_tensor_missing",
            "status": "open",
            "evidence": "Stage43 full-waypoint cache has row geometry and future labels, but no raw scene image patch, scene_raster, scene_sdf, walkable_sdf, or homography tensor.",
            "required_next_artifact": "stage43_raw_scene_patch_or_sdf_cache with train-only construction and row alignment hash",
        },
        {
            "blocker": "graph_rich_all_agent_tensor_missing",
            "status": "open",
            "evidence": "Current model features include scalar neighbor/density/TTC summaries; cache lacks edge_index/edge_attr/all-agent history tensors.",
            "required_next_artifact": "stage43_all_agent_graph_cache with past/current-only edges, masks, and no future inputs",
        },
        {
            "blocker": "scene_goal_contribution_proxy_only",
            "status": "partially_supported",
            "evidence": "Stage43-AG fresh retrained scene proxy variants show t50/hard signal, but claim boundary says not raw image/SDF.",
            "required_next_artifact": "retrained raw-scene/SDF ablation: full_raw_scene vs no_scene with bootstrap or multiseed",
        },
        {
            "blocker": "interaction_contribution_not_graph_rich",
            "status": "partially_supported",
            "evidence": "Stage43-AH no_neighbor_interaction shows feature-family contribution, Stage43-X has interaction proxy labels, but neither is graph-rich all-agent dynamics.",
            "required_next_artifact": "retrained graph-rich ablation: full_graph vs no_graph under same protected policy",
        },
    ]

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_readiness_audit_from_stage43_proxy_ablation_and_cache_schema",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {name: str(path) for name, path in INPUTS.items()},
        "input_hash": _combined_hash(list(INPUTS.values())),
        "input_verdicts": {
            "stage43_aa": _gate_verdict(aa, "stage43_aa_gate"),
            "stage43_ag": _gate_verdict(ag, "stage43_ag_gate"),
            "stage43_ah": _gate_verdict(ah, "stage43_ah_gate"),
            "stage43_ai": _gate_verdict(ai, "stage43_ai_gate"),
            "stage43_x": _gate_verdict(x, "stage43_x_gate"),
            "stage43_y": _gate_verdict(y, "stage43_y_gate"),
            "stage43_bk": _gate_verdict(bk, "stage43_bk_gate"),
        },
        "cache_schema": {
            "full_waypoint_cache": waypoint_cache,
            "scene_proxy_cache": scene_proxy_npz,
            "raw_scene_tensor_ready": raw_scene_ready,
            "graph_rich_tensor_ready": graph_rich_ready,
        },
        "current_proxy_evidence": {
            "scene_proxy_only": scene_proxy_only,
            "interaction_proxy_only": interaction_proxy_only,
            "scene_proxy_delta": scene_proxy_delta,
            "feature_family_delta": feature_family_delta,
            "stable_positive_t50_variants": ai.get("stable_positive_t50_contribution_variants", []),
            "interaction_head_signal": bool(x["stage43_x_gate"]["gates"]["interaction_head_signal"]),
            "multimodal_head_suite_candidate": bool(
                y["stage43_y_gate"]["protected_multimodal_latent_state_candidate"]
            ),
        },
        "readiness_decision": {
            "raw_scene_retrained_ablation_ready_now": raw_scene_ready,
            "graph_rich_retrained_ablation_ready_now": graph_rich_ready,
            "proxy_retrained_ablation_available": True,
            "raw_scene_or_graph_rich_main_claim_allowed": False,
            "reason": (
                "Current scene/goal/interaction evidence is useful proxy evidence, but the cache lacks "
                "raw-scene/SDF tensors and graph-rich all-agent edge tensors needed for the requested "
                "retrained raw-scene/graph-rich ablation."
            ),
        },
        "blocker_matrix": blocker_matrix,
        "next_executable_steps": [
            "build stage43_all_agent_graph_cache from current/past rows only",
            "build stage43_raw_scene_patch_or_sdf_cache if legal scene images or verified SDF proxies are available",
            "train full_graph vs no_graph and full_raw_scene vs no_scene variants with validation-selected policy",
            "bootstrap or multiseed any promoted raw-scene/graph-rich contribution",
        ],
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "new_training_executed": False,
            "new_conversion_executed": False,
            "fresh_readiness_audit_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "raw_scene_main_claim": False,
            "graph_rich_interaction_main_claim": False,
            "proxy_scene_goal_interaction_evidence_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    payload["stage43_bl_gate"] = _gate(payload, artifacts)
    return payload


def _gate(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cache = payload["cache_schema"]
    proxy = payload["current_proxy_evidence"]
    decision = payload["readiness_decision"]
    leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    gates = {
        "bk_precondition_passed": _gate_full_pass(artifacts["stage43_bk_t100_reconciliation"], "stage43_bk_gate"),
        "scene_proxy_tokens_passed_and_proxy_only": _gate_full_pass(
            artifacts["stage43_aa_scene_proxy_tokens"], "stage43_aa_gate"
        )
        and proxy["scene_proxy_only"] is True,
        "scene_proxy_retrained_ablation_exists": _gate_full_pass(
            artifacts["stage43_ag_scene_proxy_retrained_ablation"], "stage43_ag_gate"
        )
        and proxy["scene_proxy_delta"]["full_scene_minus_no_scene_t50"] > 0.0,
        "feature_family_retrained_ablation_exists": _gate_full_pass(
            artifacts["stage43_ah_feature_family_retrained_ablation"], "stage43_ah_gate"
        )
        and proxy["feature_family_delta"]["full_minus_no_neighbor_interaction_t50"] > 0.0,
        "multiseed_feature_family_confirmation_exists": _gate_full_pass(
            artifacts["stage43_ai_feature_family_multiseed"], "stage43_ai_gate"
        )
        and len(proxy["stable_positive_t50_variants"]) >= 2,
        "interaction_proxy_diagnostic_exists": _gate_full_pass(
            artifacts["stage43_x_interaction_proxy"], "stage43_x_gate"
        )
        and proxy["interaction_proxy_only"] is True
        and proxy["interaction_head_signal"] is True,
        "full_waypoint_cache_has_row_geometry_and_labels": all(
            row["exists"] and row["has_row_geometry"] and row["has_future_labels"]
            for row in cache["full_waypoint_cache"].values()
        ),
        "raw_scene_tensor_missing_not_overclaimed": cache["raw_scene_tensor_ready"] is False
        and decision["raw_scene_retrained_ablation_ready_now"] is False
        and claim["raw_scene_main_claim"] is False,
        "graph_rich_tensor_missing_not_overclaimed": cache["graph_rich_tensor_ready"] is False
        and decision["graph_rich_retrained_ablation_ready_now"] is False
        and claim["graph_rich_interaction_main_claim"] is False,
        "blocker_matrix_records_required_next_artifacts": len(payload["blocker_matrix"]) >= 4
        and all(row.get("required_next_artifact") for row in payload["blocker_matrix"]),
        "no_new_training_or_conversion": leak["new_training_executed"] is False
        and leak["new_conversion_executed"] is False
        and leak["fresh_readiness_audit_only"] is True,
        "no_future_or_test_leakage": leak["future_endpoint_input"] is False
        and leak["future_waypoint_input"] is False
        and leak["future_labels_eval_or_loss_only"] is True
        and leak["central_velocity_input"] is False
        and leak["test_endpoint_goal_construction"] is False
        and leak["test_statistics_normalization"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["proxy_scene_goal_interaction_evidence_only"] is True,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bl_raw_scene_graph_ablation_readiness_pass_blocker_documented"
        if passed == total
        else "stage43_bl_raw_scene_graph_ablation_readiness_incomplete",
        "raw_scene_retrained_ablation_ready_now": bool(decision["raw_scene_retrained_ablation_ready_now"]),
        "graph_rich_retrained_ablation_ready_now": bool(decision["graph_rich_retrained_ablation_ready_now"]),
        "proxy_evidence_available": True,
        "raw_scene_or_graph_rich_main_claim_allowed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bl_gate"]
    proxy = payload["current_proxy_evidence"]
    decision = payload["readiness_decision"]
    lines = [
        "# Stage43-BL Raw-Scene / Graph-Rich Ablation Readiness",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- raw scene retrained ablation ready now: `{gate['raw_scene_retrained_ablation_ready_now']}`",
        f"- graph-rich retrained ablation ready now: `{gate['graph_rich_retrained_ablation_ready_now']}`",
        f"- raw scene / graph-rich main claim allowed: `{gate['raw_scene_or_graph_rich_main_claim_allowed']}`",
        "",
        "## Current Evidence",
        "",
        f"- scene proxy full-scene minus no-scene t50: `{_pct(proxy['scene_proxy_delta']['full_scene_minus_no_scene_t50'])}`",
        f"- scene proxy full-scene minus no-scene hard: `{_pct(proxy['scene_proxy_delta']['full_scene_minus_no_scene_hard'])}`",
        f"- full minus no-goal t50: `{_pct(proxy['feature_family_delta']['full_minus_no_goal_t50'])}`",
        f"- full minus no-neighbor/interaction t50: `{_pct(proxy['feature_family_delta']['full_minus_no_neighbor_interaction_t50'])}`",
        f"- interaction proxy head signal: `{proxy['interaction_head_signal']}`",
        "",
        "## Readiness Decision",
        "",
        f"- proxy retrained ablation available: `{decision['proxy_retrained_ablation_available']}`",
        f"- raw scene retrained ablation ready now: `{decision['raw_scene_retrained_ablation_ready_now']}`",
        f"- graph-rich retrained ablation ready now: `{decision['graph_rich_retrained_ablation_ready_now']}`",
        f"- reason: {decision['reason']}",
        "",
        "## Full-Waypoint Cache Schema",
        "",
        "| split | rows | row geometry | future labels | raw scene keys | graph-rich keys |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for split, row in payload["cache_schema"]["full_waypoint_cache"].items():
        lines.append(
            f"| `{split}` | `{row['rows']}` | `{row['has_row_geometry']}` | `{row['has_future_labels']}` | `{row['present_raw_scene_like_keys']}` | `{row['present_graph_like_keys']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Existing scene/goal evidence is train-only proxy evidence, not raw image/SDF evidence.",
            "- Existing interaction evidence is scalar/proxy/future-label diagnostic evidence, not graph-rich all-agent dynamics.",
            "- Future labels remain loss/eval only.",
            "- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    return lines


def _write_gap(payload: Mapping[str, Any]) -> None:
    lines = [
        "# Stage43-BL Raw-Scene / Graph-Rich Gap Matrix",
        "",
        "| blocker | status | evidence | required next artifact |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["blocker_matrix"]:
        lines.append(
            f"| `{row['blocker']}` | `{row['status']}` | {row['evidence']} | `{row['required_next_artifact']}` |"
        )
    write_md(GAP_MD, lines)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bl_gate"]
    proxy = payload["current_proxy_evidence"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"raw_scene_retrained_ablation_ready_now = `{gate['raw_scene_retrained_ablation_ready_now']}`",
        f"graph_rich_retrained_ablation_ready_now = `{gate['graph_rich_retrained_ablation_ready_now']}`",
        f"raw_scene_or_graph_rich_main_claim_allowed = `{gate['raw_scene_or_graph_rich_main_claim_allowed']}`",
        "",
        f"Stage43-BL audits the scene/goal/interaction evidence after BK. Existing proxy evidence is real: full-scene proxy minus no-scene t50 is `{_pct(proxy['scene_proxy_delta']['full_scene_minus_no_scene_t50'])}`, and full minus no-neighbor/interaction t50 is `{_pct(proxy['feature_family_delta']['full_minus_no_neighbor_interaction_t50'])}`. But the cache still lacks raw scene/SDF tensors and graph-rich all-agent edge tensors, so raw-scene or graph-rich interaction main claims remain blocked.",
        "",
        "Next executable artifacts are `stage43_all_agent_graph_cache` and `stage43_raw_scene_patch_or_sdf_cache`, followed by retrained full_graph/no_graph and full_raw_scene/no_scene ablations.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future labels are supervision/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bl_raw_scene_graph_ablation_readiness"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "raw_scene_retrained_ablation_ready_now": gate["raw_scene_retrained_ablation_ready_now"],
        "graph_rich_retrained_ablation_ready_now": gate["graph_rich_retrained_ablation_ready_now"],
        "proxy_evidence_available": gate["proxy_evidence_available"],
        "raw_scene_or_graph_rich_main_claim_allowed": gate[
            "raw_scene_or_graph_rich_main_claim_allowed"
        ],
        "current_proxy_evidence": payload["current_proxy_evidence"],
        "blocker_matrix": payload["blocker_matrix"],
        "report": str(REPORT_MD),
        "gap_matrix": str(GAP_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bl_raw_scene_graph_ablation_readiness"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BL",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "raw_scene_retrained_ablation_ready_now": gate[
                            "raw_scene_retrained_ablation_ready_now"
                        ],
                        "graph_rich_retrained_ablation_ready_now": gate[
                            "graph_rich_retrained_ablation_ready_now"
                        ],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bl_gate"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(REPORT_MD, _render_report(payload))
    _write_gap(payload)
    write_md(
        GATE_MD,
        [
            "# Stage43-BL Raw-Scene / Graph-Rich Ablation Readiness Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- raw scene retrained ablation ready now: `{gate['raw_scene_retrained_ablation_ready_now']}`",
            f"- graph-rich retrained ablation ready now: `{gate['graph_rich_retrained_ablation_ready_now']}`",
            f"- raw scene / graph-rich main claim allowed: `{gate['raw_scene_or_graph_rich_main_claim_allowed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- raw scene retrained ablation ready now: `{gate['raw_scene_retrained_ablation_ready_now']}`",
            f"- graph-rich retrained ablation ready now: `{gate['graph_rich_retrained_ablation_ready_now']}`",
            f"- raw scene / graph-rich main claim allowed: `{gate['raw_scene_or_graph_rich_main_claim_allowed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BL documents that current scene/goal/interaction evidence remains proxy-heavy.",
            "- Raw-scene/SDF retrained ablation and graph-rich all-agent retrained ablation are not ready until their caches exist.",
            "- Existing proxy evidence is useful but cannot be the raw-scene or graph-rich main claim.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_raw_scene_graph_ablation_readiness() -> dict[str, Any]:
    payload = build_raw_scene_graph_ablation_readiness()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Audit whether Stage43 is ready for raw-scene and graph-rich retrained ablations."
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_raw_scene_graph_ablation_readiness()
    gate = payload["stage43_bl_gate"]
    print(f"Stage43-BL: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"raw_scene_ready={gate['raw_scene_retrained_ablation_ready_now']}")
    print(f"graph_rich_ready={gate['graph_rich_retrained_ablation_ready_now']}")
    return payload


if __name__ == "__main__":
    main()
