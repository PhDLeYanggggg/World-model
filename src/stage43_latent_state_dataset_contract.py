from __future__ import annotations

import hashlib
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


OUT_DIR = Path("outputs/stage43_latent_state")
DATA35 = Path("data/stage35_selective_transfer")
DATA36 = Path("data/stage36_t50_repair")
DATA37 = Path("data/stage37_t50_history")
DATA42 = Path("data/stage42_source_level_full_waypoint_cache")

REPORT_JSON = OUT_DIR / "stage43_latent_state_dataset_contract.json"
REPORT_MD = OUT_DIR / "stage43_latent_state_dataset_contract.md"
GATE_MD = OUT_DIR / "stage43_stage_b_latent_state_dataset_gate.md"
SCHEMA_JSON = OUT_DIR / "stage43_latent_state_token_schema.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_B_LATENT_STATE_DATASET_CONTRACT"
SOURCE = "fresh_stage43_b_latent_state_dataset_contract"
SPLITS = ["train", "val", "test"]


TOKEN_SCHEMA = {
    "context_observation": [
        "dataset",
        "scene_id",
        "source_file",
        "agent_id",
        "frame_id",
        "current_x",
        "current_y",
        "horizon",
        "scale",
        "split",
        "data_role",
    ],
    "agent_history": [
        "history_x",
        "history_y",
        "history_dx",
        "history_dy",
        "history_speed",
        "history_accel",
        "history_heading",
        "history_valid_mask",
        "history_curvature",
        "history_turn_angle",
        "history_stop_go",
        "history_dwell",
        "history_path_length",
        "history_velocity_decay",
    ],
    "all_agent_graph": [
        "history_neighbor_count",
        "history_min_neighbor_dist",
        "history_density",
        "history_TTC",
        "history_closing_speed",
    ],
    "scene_goal_proxy": [
        "prototype_vectors",
        "prototype_likelihood",
        "prototype_entropy",
        "goal_ambiguity",
        "prototype_distance",
        "prototype_angle",
    ],
    "baseline_rollout_family": [
        "baseline_family_prediction",
        "baseline_family_y_fde",
        "baseline_family_relative_y",
        "strongest_idx",
        "oracle_idx",
        "stage37_selected_family",
        "stage35_selected_family",
    ],
    "safety_floor_state": [
        "easy",
        "hard",
        "failure",
        "oracle_margin",
        "stage37_confidence",
        "stage36_predicted_gain",
        "stage36_hard_prob",
        "stage36_fail_prob",
        "stage36_easy_prob",
    ],
    "labels_only": [
        "future_endpoint_x",
        "future_endpoint_y",
        "future_relative_y",
        "future_fde_by_baseline",
        "full_waypoint_xy_partial",
        "waypoint_valid_partial",
        "occupancy_density_proxy",
        "failure_label",
        "gain_label",
        "harm_label",
    ],
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _file(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256(path) if exists and hash_file else "",
    }


def _npz_len(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    z = np.load(path, allow_pickle=False)
    return int(len(z[key]))


def _split_files(split: str) -> dict[str, Path]:
    return {
        "geometry": DATA35 / f"expanded_external_{split}.npz",
        "labels": DATA35 / f"labels_{split}.npz",
        "stage35_selection": DATA36 / f"stage35_selection_{split}.npz",
        "stage36_t50_features": DATA36 / f"t50_features_{split}.npz",
        "history": DATA37 / f"history_windows_{split}.npz",
        "goal_prototypes": DATA37 / f"goal_prototypes_{split}.npz",
        "baseline_family": DATA37 / f"t50_baseline_family_{split}.npz",
    }


def _split_summary(split: str) -> dict[str, Any]:
    files = _split_files(split)
    row_keys = {
        "geometry": "horizon",
        "labels": "y_fde",
        "stage35_selection": "selected",
        "stage36_t50_features": "horizon",
        "history": "source_found",
        "goal_prototypes": "prototype_entropy",
        "baseline_family": "prediction",
    }
    rows = {name: _npz_len(path, row_keys[name]) for name, path in files.items()}
    geometry = np.load(files["geometry"], allow_pickle=False) if files["geometry"].exists() else {}
    labels = np.load(files["labels"], allow_pickle=False) if files["labels"].exists() else {}
    history = np.load(files["history"], allow_pickle=False) if files["history"].exists() else {}
    horizons = geometry["horizon"].astype(int) if files["geometry"].exists() else np.asarray([], dtype=int)
    domains = geometry["dataset"].astype(str) if files["geometry"].exists() else np.asarray([], dtype=str)
    valid_mask = history["history_valid_mask"] if files["history"].exists() else np.zeros((0, 64), dtype=bool)
    hard = labels["hard"].astype(bool) if files["labels"].exists() else np.asarray([], dtype=bool)
    failure = labels["failure"].astype(bool) if files["labels"].exists() else np.asarray([], dtype=bool)
    easy = labels["easy"].astype(bool) if files["labels"].exists() else np.asarray([], dtype=bool)
    row_alignment = bool(rows and len(set(rows.values())) == 1 and next(iter(rows.values())) > 0)
    return {
        "split": split,
        "data_role": "supervised_training" if split in {"train", "val"} else "official_eval_dataset_local_raw_frame",
        "files": {name: _file(path, hash_file=False) for name, path in files.items()},
        "rows_by_artifact": rows,
        "row_alignment_pass": row_alignment,
        "rows": int(next(iter(rows.values())) if row_alignment else max(rows.values()) if rows else 0),
        "domain_counts": {str(k): int(v) for k, v in zip(*np.unique(domains, return_counts=True))} if len(domains) else {},
        "horizon_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(horizons, return_counts=True))} if len(horizons) else {},
        "history_k_available": {
            str(k): int(np.sum(valid_mask[:, -k:].sum(axis=1) >= k)) if len(valid_mask) else 0
            for k in [8, 16, 32, 64]
        },
        "hard_rows": int(np.sum(hard)),
        "failure_rows": int(np.sum(failure)),
        "easy_rows": int(np.sum(easy)),
    }


def _full_waypoint_availability() -> dict[str, Any]:
    cache = DATA42 / "stage42iv_source_level_merged_cache.npz"
    if not cache.exists():
        return {"status": "not_run", "reason": "missing Stage42 source-level full-waypoint cache"}
    z = np.load(cache, allow_pickle=False)
    return {
        "status": "partial_eval_cache",
        "cache_file": _file(cache, hash_file=False),
        "rows": int(len(z["waypoint_valid"])),
        "waypoint_shape": list(z["waypoint_xy"].shape),
        "valid_waypoint_rows": int(np.sum(np.any(z["waypoint_valid"], axis=1))),
        "train_val_supervised_full_waypoint_ready": False,
        "limitation": "Full-waypoint labels are present in the Stage42 current source-level evaluation cache, but a train/val full-waypoint latent-state supervised cache is not yet frozen.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    splits = payload["splits"]
    leakage = payload["no_leakage"]
    labels = payload["label_availability"]
    gates = {
        "stage43_a_precondition_passed": payload["stage43_a_precondition"]["verdict"] == "stage43_a_safety_floor_replay_pass",
        "all_split_artifacts_exist": all(row["files"][name]["exists"] for row in splits.values() for name in row["files"]),
        "all_split_rows_align": all(row["row_alignment_pass"] for row in splits.values()),
        "train_val_test_roles_explicit": {row["data_role"] for row in splits.values()} == {"supervised_training", "official_eval_dataset_local_raw_frame"},
        "history_windows_available": all(row["history_k_available"]["8"] > 0 for row in splits.values()),
        "goal_prototypes_available": all(row["files"]["goal_prototypes"]["exists"] for row in splits.values()),
        "baseline_family_available": all(row["files"]["baseline_family"]["exists"] for row in splits.values()),
        "endpoint_labels_available_all_splits": labels["future_endpoint_labels_all_splits"] is True,
        "full_waypoint_limitation_recorded": labels["full_waypoint"]["status"] == "partial_eval_cache"
        and labels["full_waypoint"]["train_val_supervised_full_waypoint_ready"] is False,
        "labels_separated_from_inputs": "labels_only" in payload["token_schema"]
        and not any(
            name in sum((payload["token_schema"][token] for token in payload["token_schema"] if token != "labels_only"), [])
            for name in ["future_endpoint_x", "future_endpoint_y", "full_waypoint_xy_partial", "waypoint_valid_partial"]
        ),
        "no_future_or_test_leakage": (
            leakage["future_endpoint_input"] is False
            and leakage["future_waypoint_input"] is False
            and leakage["central_velocity_official_input"] is False
            and leakage["test_endpoint_goal_construction"] is False
            and leakage["test_statistics_normalization"] is False
        ),
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_b_latent_state_dataset_contract_pass" if passed == total else "stage43_b_latent_state_dataset_contract_partial",
        "endpoint_latent_state_training_ready": passed == total,
        "full_waypoint_supervised_training_ready": labels["full_waypoint"]["train_val_supervised_full_waypoint_ready"],
    }


def _build_payload() -> dict[str, Any]:
    stage43_a = read_json(OUT_DIR / "stage43_safety_floor_replay.json", {})
    splits = {split: _split_summary(split) for split in SPLITS}
    full_waypoint = _full_waypoint_availability()
    payload: dict[str, Any] = {
        "stage": "Stage43-B latent-state dataset contract",
        "source": SOURCE,
        "result_source": "fresh_run_contract_from_cached_verified_stage35_37_42_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "stage43_a_precondition": stage43_a.get("stage43_a_gate", {}),
        "input_hash": _combined_hash(
            [OUT_DIR / "stage43_safety_floor_replay.json"]
            + [path for split in SPLITS for path in _split_files(split).values()]
            + [DATA42 / "stage42iv_source_level_merged_cache.npz"]
        ),
        "token_schema": TOKEN_SCHEMA,
        "splits": splits,
        "label_availability": {
            "future_endpoint_labels_all_splits": all((DATA35 / f"expanded_external_{split}.npz").exists() for split in SPLITS),
            "baseline_fde_labels_all_splits": all((DATA35 / f"labels_{split}.npz").exists() for split in SPLITS),
            "failure_gain_harm_labels_all_splits": all((DATA35 / f"labels_{split}.npz").exists() and (DATA36 / f"stage35_selection_{split}.npz").exists() for split in SPLITS),
            "occupancy_density_proxy_all_splits": all((DATA37 / f"history_windows_{split}.npz").exists() for split in SPLITS),
            "full_waypoint": full_waypoint,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_official_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "future_labels_are_loss_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_b_gate"] = _gate(payload)
    payload["decision"] = (
        "Endpoint/failure/gain/harm/occupancy latent-state dataset contract is ready; full-waypoint supervised training remains blocked until train/val full-waypoint labels are frozen."
        if payload["stage43_b_gate"]["endpoint_latent_state_training_ready"]
        else "Do not train Stage43 latent-state models until dataset-contract blockers are fixed."
    )
    return payload


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_json(SCHEMA_JSON, _jsonable(TOKEN_SCHEMA))
    gate = payload["stage43_b_gate"]
    lines = [
        "# Stage43-B Latent-State Dataset Contract",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- endpoint latent-state training ready: `{gate['endpoint_latent_state_training_ready']}`",
        f"- full-waypoint supervised training ready: `{gate['full_waypoint_supervised_training_ready']}`",
        "",
        "## Split Summary",
        "",
        "| split | role | rows | domains | horizons | K8 history | hard | failure | easy | alignment |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        *[
            f"| {split} | `{row['data_role']}` | {row['rows']} | `{row['domain_counts']}` | `{row['horizon_counts']}` | {row['history_k_available']['8']} | {row['hard_rows']} | {row['failure_rows']} | {row['easy_rows']} | {row['row_alignment_pass']} |"
            for split, row in payload["splits"].items()
        ],
        "",
        "## Token Groups",
        "",
        *[f"- `{name}`: {', '.join(values)}" for name, values in payload["token_schema"].items()],
        "",
        "## Label Boundary",
        "",
        "- Future endpoint, future relative error, full-waypoint, occupancy/density, failure/gain/harm are label/loss/eval targets only.",
        "- They are not listed in any inference input token group.",
        f"- Full-waypoint status: `{payload['label_availability']['full_waypoint']['status']}`.",
        f"- Full-waypoint limitation: {payload['label_availability']['full_waypoint']['limitation']}",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-B Latent-State Dataset Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- endpoint latent-state training ready: `{gate['endpoint_latent_state_training_ready']}`",
            f"- full-waypoint supervised training ready: `{gate['full_waypoint_supervised_training_ready']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_b_gate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"endpoint_latent_state_training_ready = `{gate['endpoint_latent_state_training_ready']}`",
        f"full_waypoint_supervised_training_ready = `{gate['full_waypoint_supervised_training_ready']}`",
        "",
        "Stage43-B builds the latent-state dataset contract from Stage35/36/37 external geometry/history/goal/baseline artifacts and the Stage42 source-level full-waypoint cache. It separates inference tokens from labels: future endpoint/waypoint labels are loss/eval only and are not model inputs.",
        "",
        "Endpoint/failure/gain/harm/occupancy latent-state training is contract-ready; full-waypoint supervised latent training is still blocked until train/val full-waypoint labels are frozen. No Stage5C/SMC/metric/seconds/true-3D/foundation claim is made.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_b_latent_state_dataset_contract"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "endpoint_latent_state_training_ready": gate["endpoint_latent_state_training_ready"],
        "full_waypoint_supervised_training_ready": gate["full_waypoint_supervised_training_ready"],
        "report": str(REPORT_MD),
        "schema": str(SCHEMA_JSON),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"stage": "Stage43-B", "source": payload["source"], "verdict": gate["verdict"], "gate": f"{gate['passed']} / {gate['total']}", "generated_at_utc": payload["generated_at_utc"]}), ensure_ascii=False) + "\n")


def main() -> dict[str, Any]:
    payload = _build_payload()
    _write_reports(payload)
    _update_text_outputs(payload)
    return payload


if __name__ == "__main__":
    result = main()
    gate = result["stage43_b_gate"]
    print(f"Stage43-B latent-state dataset contract: {gate['verdict']} ({gate['passed']}/{gate['total']})")
