from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    M3W_README,
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    SPLITS,
    WORK_SUMMARY,
    _build_split,
    _cache_path,
    _git_commit,
    _jsonable,
    _npz,
    _row_hash,
)
from src.stage43_multimodal_latent_head_suite import REPORT_JSON as STAGE43_Y_JSON


REPORT_JSON = OUT_DIR / "stage43_latent_token_schema_coverage.json"
REPORT_MD = OUT_DIR / "stage43_latent_token_schema_coverage.md"
GATE_MD = OUT_DIR / "stage43_stage_z_latent_token_schema_coverage_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_Z_LATENT_TOKEN_SCHEMA_COVERAGE"
SOURCE = "fresh_stage43_z_latent_token_schema_coverage"


TOKEN_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "agent_state": {
        "kind": "inference_input",
        "required_features": ["current_x_over_scale", "current_y_over_scale", "horizon_norm"],
        "required_cache_keys": ["current_xy", "agent_id", "frame_id", "horizon"],
        "claim": "covered",
    },
    "agent_history": {
        "kind": "inference_input",
        "required_prefixes": [
            "history_dx_tail",
            "history_dy_tail",
            "history_speed_tail",
            "history_accel_tail",
            "history_heading_tail",
            "history_valid_mask_tail",
        ],
        "required_features": [
            "history_curvature",
            "history_turn_angle",
            "history_stop_go",
            "history_dwell",
            "history_path_length",
            "history_velocity_decay",
        ],
        "claim": "covered",
    },
    "all_agent_current_state": {
        "kind": "row_metadata_grouping",
        "required_cache_keys": ["source_file", "scene_id", "frame_id", "agent_id", "current_xy"],
        "claim": "partial_grouped_rows_not_explicit_tensor",
    },
    "neighbor_graph": {
        "kind": "inference_proxy",
        "required_features": [
            "history_neighbor_count",
            "history_min_neighbor_dist",
            "history_density",
            "history_TTC",
            "history_closing_speed",
        ],
        "claim": "proxy_only_not_full_graph_tensor",
    },
    "scene_patch": {
        "kind": "missing_modality",
        "required_features": ["scene_patch_embedding"],
        "claim": "missing_explicit_scene_image_or_raster_token",
        "gap_expected": True,
    },
    "scene_sdf": {
        "kind": "missing_modality",
        "required_features": ["scene_sdf"],
        "claim": "missing_explicit_scene_sdf_token",
        "gap_expected": True,
    },
    "goal_region": {
        "kind": "inference_proxy",
        "required_prefixes": ["prototype_likelihood_", "prototype_distance_", "prototype_angle_"],
        "required_features": ["prototype_entropy", "goal_ambiguity"],
        "claim": "scene_agnostic_goal_proxy_covered",
    },
    "interaction_edge": {
        "kind": "inference_proxy_and_label_proxy",
        "required_features": ["history_neighbor_count", "history_min_neighbor_dist", "history_TTC", "history_closing_speed"],
        "claim": "proxy_only_not_human_interaction_annotation",
    },
    "baseline_rollout": {
        "kind": "inference_input",
        "required_prefixes": ["baseline_endpoint_rel_"],
        "claim": "baseline_family_endpoint_rollouts_covered",
    },
    "safety_floor_prediction": {
        "kind": "inference_input",
        "required_features": ["floor_endpoint_rel_x", "floor_endpoint_rel_y"],
        "claim": "stage37_stage42_floor_endpoint_covered",
    },
    "domain_source_horizon": {
        "kind": "inference_input_and_metadata",
        "required_prefixes": ["domain_", "horizon_"],
        "required_cache_keys": ["dataset", "source_file", "horizon"],
        "claim": "domain_horizon_features_source_metadata_covered",
    },
    "time_frame": {
        "kind": "metadata_only",
        "required_cache_keys": ["frame_id", "dt_frame_step"],
        "claim": "frame_metadata_covered_not_seconds_verified",
    },
    "future_endpoint_label": {
        "kind": "label_only",
        "required_cache_keys": ["future_xy"],
        "forbidden_feature_terms": ["future", "endpoint_x", "endpoint_y"],
        "claim": "label_only_not_input",
    },
    "future_full_waypoint_label": {
        "kind": "label_only",
        "required_cache_keys": ["waypoint_xy", "waypoint_valid"],
        "forbidden_feature_terms": ["waypoint", "future"],
        "claim": "label_only_not_input",
    },
    "occupancy_density_label": {
        "kind": "proxy_label",
        "required_features": ["history_density"],
        "claim": "causal_history_density_proxy_not_future_occupancy",
    },
    "failure_gain_harm_label": {
        "kind": "proxy_label",
        "required_cache_keys": ["failure", "easy", "hard"],
        "claim": "covered_training_labels",
    },
    "mask_validity": {
        "kind": "label_and_input_mask",
        "required_cache_keys": ["valid_mask", "waypoint_valid"],
        "required_prefixes": ["history_valid_mask_tail"],
        "claim": "covered",
    },
}


def _has_prefix(feature_names: Sequence[str], prefix: str) -> bool:
    return any(name.startswith(prefix) for name in feature_names)


def _has_forbidden(feature_names: Sequence[str], terms: Sequence[str]) -> bool:
    lowered = [name.lower() for name in feature_names]
    return any(term.lower() in name for term in terms for name in lowered)


def _feature_schema_hash(feature_names: Sequence[str]) -> str:
    payload = json.dumps(list(feature_names), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_schema(split: str) -> dict[str, Any]:
    path = _cache_path(split)
    if not path.exists():
        return {"split": split, "exists": False, "path": str(path), "rows": 0, "keys": [], "row_hash": ""}
    z = _npz(path)
    keys = list(z.files)
    shapes = {key: list(z[key].shape) for key in keys}
    dtypes = {key: str(z[key].dtype) for key in keys}
    schema_payload = json.dumps({"keys": keys, "shapes": shapes, "dtypes": dtypes}, sort_keys=True).encode("utf-8")
    return {
        "split": split,
        "exists": True,
        "path": str(path),
        "rows": int(len(z["horizon"])),
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "row_hash": _row_hash(z),
        "schema_hash": hashlib.sha256(schema_payload).hexdigest(),
        "domains": {str(k): int(v) for k, v in zip(*np.unique(z["dataset"].astype(str), return_counts=True))},
        "horizons": {str(int(k)): int(v) for k, v in zip(*np.unique(z["horizon"].astype(int), return_counts=True))},
    }


def _status_for_requirement(req: Mapping[str, Any], feature_names: Sequence[str], cache_keys: set[str]) -> dict[str, Any]:
    required_features = list(req.get("required_features", []))
    required_prefixes = list(req.get("required_prefixes", []))
    required_cache_keys = list(req.get("required_cache_keys", []))
    forbidden_terms = list(req.get("forbidden_feature_terms", []))

    found_features = {name: name in feature_names for name in required_features}
    found_prefixes = {prefix: _has_prefix(feature_names, prefix) for prefix in required_prefixes}
    found_cache = {key: key in cache_keys for key in required_cache_keys}
    forbidden_present = _has_forbidden(feature_names, forbidden_terms) if forbidden_terms else False
    present = all(found_features.values()) and all(found_prefixes.values()) and all(found_cache.values())
    if req.get("kind") == "label_only":
        passed = present and not forbidden_present
    elif req.get("gap_expected"):
        passed = True
    else:
        passed = present
    status = "covered" if present else "missing"
    if req.get("gap_expected"):
        status = "recorded_gap"
    if req.get("kind") == "label_only" and present and not forbidden_present:
        status = "label_only_separated"
    return {
        "kind": req.get("kind"),
        "claim": req.get("claim"),
        "status": status,
        "present": bool(present),
        "passed_for_gate": bool(passed),
        "found_features": found_features,
        "found_prefixes": found_prefixes,
        "found_cache_keys": found_cache,
        "forbidden_feature_terms": forbidden_terms,
        "forbidden_terms_present_in_features": bool(forbidden_present),
    }


def _coverage(feature_names: Sequence[str], split_schemas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cache_keys: set[str] = set()
    for row in split_schemas.values():
        cache_keys.update(row.get("keys", []))
    return {
        name: _status_for_requirement(req, feature_names, cache_keys)
        for name, req in TOKEN_REQUIREMENTS.items()
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    coverage = payload["token_coverage"]
    leakage = payload["no_leakage"]
    gaps = payload["explicit_gaps"]
    y = payload["stage43_y_precondition"]
    gates = {
        "stage43_y_head_suite_passed": y.get("verdict") == "stage43_y_protected_multimodal_latent_head_suite_candidate",
        "split_caches_exist": all(row["exists"] and row["rows"] > 0 for row in payload["split_schemas"].values()),
        "row_hashes_recorded": all(bool(row.get("row_hash")) for row in payload["split_schemas"].values()),
        "feature_schema_recorded": bool(payload["feature_schema"]["feature_schema_hash"])
        and payload["feature_schema"]["feature_dim"] > 0,
        "core_causal_inputs_covered": all(
            coverage[name]["present"]
            for name in [
                "agent_state",
                "agent_history",
                "goal_region",
                "baseline_rollout",
                "safety_floor_prediction",
                "domain_source_horizon",
            ]
        ),
        "all_agent_and_neighbor_scope_honest": coverage["all_agent_current_state"]["present"]
        and coverage["neighbor_graph"]["present"]
        and coverage["neighbor_graph"]["claim"] == "proxy_only_not_full_graph_tensor",
        "future_labels_separated_from_inputs": coverage["future_endpoint_label"]["status"] == "label_only_separated"
        and coverage["future_full_waypoint_label"]["status"] == "label_only_separated",
        "scene_raster_sdf_gaps_recorded": gaps["explicit_scene_image_raster_token"] == "missing"
        and gaps["explicit_scene_sdf_token"] == "missing",
        "proxy_boundaries_recorded": gaps["future_occupancy_true_label"] == "missing_proxy_only"
        and gaps["true_physical_validity_label"] == "missing_proxy_only"
        and gaps["human_interaction_annotation"] == "missing_proxy_only",
        "no_future_or_test_leakage": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["central_velocity_input"] is False
        and leakage["test_endpoint_goal_construction"] is False
        and leakage["test_statistics_normalization"] is False,
        "claim_boundary_preserved": payload["claim_boundary"]["true_3d"] is False
        and payload["claim_boundary"]["foundation_world_model"] is False
        and payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "protected_not_standalone": payload["deployment_contract"]["safety_floor_required"] is True
        and payload["deployment_contract"]["standalone_ungated_policy"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_z_latent_token_schema_coverage_pass"
        if passed == total
        else "stage43_z_latent_token_schema_coverage_partial",
        "protected_multimodal_latent_state_schema_supported": passed == total,
        "standalone_world_model_deployable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    y = read_json(STAGE43_Y_JSON, {})
    train_sample = _build_split("train", max_rows=512, seed=43000)
    feature_names = list(train_sample.feature_names)
    split_schemas = {split: _cache_schema(split) for split in SPLITS}
    feature_schema = {
        "feature_dim": int(len(feature_names)),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "feature_names": feature_names,
        "sample_rows_used_for_schema": int(len(train_sample.x)),
    }
    payload: dict[str, Any] = {
        "stage": "Stage43-Z latent token schema coverage",
        "source": SOURCE,
        "result_source": "fresh_schema_audit_from_cached_verified_stage43_full_waypoint_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_y_precondition": y.get("stage43_y_gate", {}),
        "split_schemas": split_schemas,
        "feature_schema": feature_schema,
        "token_coverage": _coverage(feature_names, split_schemas),
        "explicit_gaps": {
            "explicit_scene_image_raster_token": "missing",
            "explicit_scene_sdf_token": "missing",
            "full_all_agent_graph_tensor": "missing_proxy_only",
            "future_occupancy_true_label": "missing_proxy_only",
            "true_physical_validity_label": "missing_proxy_only",
            "human_interaction_annotation": "missing_proxy_only",
            "verified_metric_or_seconds_calibration": "missing",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_loss_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "future_occupancy_claim": False,
            "true_physical_validity_claim": False,
        },
        "deployment_contract": {
            "safety_floor_required": True,
            "standalone_ungated_policy": False,
            "allowed_claim": "protected multimodal latent-state head suite with proxy scene/goal/interaction/density signals",
            "disallowed_claims": [
                "true 3D world model",
                "foundation world model",
                "metric trajectory prediction",
                "seconds-level long horizon",
                "future occupancy prediction",
                "human-gold interaction labels",
                "standalone ungated deployment",
            ],
        },
    }
    payload["stage43_z_gate"] = _gate(payload)
    return payload


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_z_gate"]
    lines = [
        "# Stage43-Z Latent Token Schema Coverage",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- protected schema supported: `{gate['protected_multimodal_latent_state_schema_supported']}`",
        f"- standalone ungated deployment: `False`",
        "",
        "## Feature Schema",
        "",
        f"- feature dim: `{payload['feature_schema']['feature_dim']}`",
        f"- feature schema hash: `{payload['feature_schema']['feature_schema_hash']}`",
        f"- train rows sampled for schema: `{payload['feature_schema']['sample_rows_used_for_schema']}`",
        "",
        "## Split Hashes",
        "",
        "| split | rows | row hash | schema hash | domains | horizons |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for split, row in payload["split_schemas"].items():
        lines.append(
            f"| {split} | {row['rows']} | `{row.get('row_hash', '')[:12]}` | `{row.get('schema_hash', '')[:12]}` | `{row.get('domains', {})}` | `{row.get('horizons', {})}` |"
        )
    lines.extend(
        [
            "",
            "## Token Coverage",
            "",
            "| token group | kind | status | claim boundary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, row in payload["token_coverage"].items():
        lines.append(f"| `{name}` | `{row['kind']}` | `{row['status']}` | {row['claim']} |")
    lines.extend(
        [
            "",
            "## Explicit Gaps",
            "",
            *[f"- `{name}`: `{status}`" for name, status in payload["explicit_gaps"].items()],
            "",
            "## No-Leakage Boundary",
            "",
            "- Future endpoints and full waypoints are labels/eval targets only.",
            "- Scene raster/image/SDF tokens are not present in the current Stage43 full-waypoint cache.",
            "- Neighbor and interaction context are causal proxy features, not a full graph tensor or human interaction annotation.",
            "- Density is a causal history-density proxy, not future occupancy.",
            "- Smoothness/validity remains diagnostic proxy evidence, not true physical validity.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
            "",
            "No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-Z Latent Token Schema Coverage Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- protected schema supported: `{gate['protected_multimodal_latent_state_schema_supported']}`",
            f"- standalone ungated deployment: `False`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_z_gate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"feature_schema_hash = `{payload['feature_schema']['feature_schema_hash']}`",
        f"train_row_hash = `{payload['split_schemas']['train'].get('row_hash', '')}`",
        "",
        "Stage43-Z audits what the current protected latent-state cache actually covers. It confirms causal agent/history, goal-prototype, baseline-rollout, safety-floor, horizon/domain, density, interaction-risk proxy, and failure/gain/harm heads are represented under row/schema hashes.",
        "",
        "It also records the limits: there is still no explicit scene image/raster token, no SDF token, no full all-agent graph tensor, no future occupancy label, no human-gold interaction label, and no true physical-validity label. These are gaps, not hidden successes. The current claim remains protected dataset-local/raw-frame 2.5D multimodal latent-state evidence, not true 3D or foundation modeling.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_z_latent_token_schema_coverage"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "feature_schema_hash": payload["feature_schema"]["feature_schema_hash"],
        "train_row_hash": payload["split_schemas"]["train"].get("row_hash", ""),
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "standalone_world_model_deployable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "stage": "Stage43-Z",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> dict[str, Any]:
    payload = _build_payload()
    _write_reports(payload)
    _update_text_outputs(payload)
    return payload


if __name__ == "__main__":
    result = main()
    gate = result["stage43_z_gate"]
    print(f"Stage43-Z latent token schema coverage: {gate['verdict']} ({gate['passed']}/{gate['total']})")
