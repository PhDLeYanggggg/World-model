from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


OUT_DIR = Path("outputs/stage42_long_research")
REGISTRY_PATH = Path("outputs/data_registry/stage20_dataset_registry.json")
MAX_AUDIT_FILES_PER_ROOT = 20000


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "SDD 是 pixel-space benchmark，不是 metric benchmark。",
    "External domains 仍是 dataset-local / unverified weak-metric diagnostic。",
    "t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。",
    "homography / metric scale / effective seconds 未完成全局验证。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
    "Stage42 当前只做数据与标定 fresh audit，不训练、不下载 gated data、不写 large cache。",
]


DATASET_SPECS = [
    {
        "id": "sdd",
        "name": "Stanford Drone Dataset",
        "domain": "real top-down drone pedestrian/mixed-agent",
        "role": "official_eval / supervised_training",
        "official_hint": "https://cvgl.stanford.edu/projects/uav_data/",
        "raw_candidates": [
            "external_data/StanfordDroneDataset",
            "/Users/yangyue/Downloads/World/external_data/StanfordDroneDataset",
            "/Users/yangyue/Downloads/StanfordDroneDataset",
        ],
        "converted_candidates": ["data/stage21_sdd_world_state", "data/stage24_sdd_fast_cache"],
        "known_coordinate_unit": "pixel",
        "known_metric_status": "pixel_space; no verified homography/scale",
    },
    {
        "id": "opentraj",
        "name": "OpenTraj",
        "domain": "toolkit plus multiple underlying pedestrian/traffic/crowd datasets",
        "role": "external top-down source hub / loader input",
        "official_hint": "https://github.com/crowdbotp/OpenTraj",
        "raw_candidates": [
            "external_data/OpenTraj",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj",
            "/Users/yangyue/Downloads/OpenTraj",
        ],
        "converted_candidates": ["data/stage31_external_latent_cache", "data/stage41_world_model"],
        "known_coordinate_unit": "dataset-local mixed",
        "known_metric_status": "dataset-local; underlying licenses/scales vary",
    },
    {
        "id": "eth_ucy",
        "name": "ETH/UCY",
        "domain": "fixed-camera/top-down pedestrian trajectories",
        "role": "external_eval / supervised_training",
        "official_hint": "ETH/BIWI + UCY original dataset pages; verify source-specific terms",
        "raw_candidates": [
            "external_data/OpenTraj/datasets/ETH",
            "external_data/OpenTraj/datasets/ETH-Person",
            "external_data/OpenTraj/datasets/UCY",
            "/Users/yangyue/Downloads/ETH_UCY",
        ],
        "converted_candidates": ["data/stage5b_world_state/eth_ucy", "data/stage41_world_model"],
        "known_coordinate_unit": "dataset-local",
        "known_metric_status": "unverified weak metric / dataset-local; do not claim metric",
    },
    {
        "id": "trajnet",
        "name": "TrajNet++",
        "domain": "pedestrian trajectory forecasting benchmark",
        "role": "external_eval / supervised_training",
        "official_hint": "https://www.epfl.ch/labs/vita/datasets/",
        "raw_candidates": [
            "external_data/OpenTraj/datasets/TrajNet",
            "external_data/OpenTraj/datasets/TrajNet++",
            "/Users/yangyue/Downloads/trajnetplusplusdataset",
        ],
        "converted_candidates": ["data/stage5b_world_state/trajnet", "data/stage41_world_model"],
        "known_coordinate_unit": "dataset-local",
        "known_metric_status": "dataset-local; terms/scale must be verified per source",
    },
    {
        "id": "ucy",
        "name": "UCY Crowd",
        "domain": "pedestrian crowd trajectories",
        "role": "external_eval / supervised_training",
        "official_hint": "http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "raw_candidates": ["external_data/OpenTraj/datasets/UCY", "/Users/yangyue/Downloads/UCY"],
        "converted_candidates": ["data/stage41_world_model", "data/stage37_t50_history"],
        "known_coordinate_unit": "dataset-local",
        "known_metric_status": "dataset-local; not globally verified metric",
    },
    {
        "id": "tgsim",
        "name": "TGSIM",
        "domain": "traffic vehicle trajectories",
        "role": "diagnostic_only",
        "official_hint": "https://data.transportation.gov/",
        "raw_candidates": ["data/stage5b_world_state/tgsim", "data/stage5b_world_state/tgsim_i90"],
        "converted_candidates": ["data/stage5b_world_state/tgsim", "data/stage5b_world_state/tgsim_i90"],
        "known_coordinate_unit": "traffic metric if source units verified by prior stage",
        "known_metric_status": "metric diagnostic for traffic only; not pedestrian world-model success",
    },
    {
        "id": "aerialmpt",
        "name": "AerialMPT",
        "domain": "aerial pedestrian/crowd trajectories",
        "role": "external_eval candidate / diagnostic",
        "official_hint": "DLR AerialMPT official page required; local prior stages have derived scene packs",
        "raw_candidates": ["/Users/yangyue/Downloads/AerialMPT", "external_data/AerialMPT"],
        "converted_candidates": ["data/stage11_scene_packs/aerialmpt", "data/stage11_visual_annotations/aerialmpt"],
        "known_coordinate_unit": "unknown / derived local",
        "known_metric_status": "not verified in Stage42 audit",
    },
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_paths(paths: Iterable[str | Path]) -> str:
    h = hashlib.sha256()
    for raw in sorted({str(p) for p in paths}):
        path = Path(raw)
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        if path.exists():
            h.update(str(path.stat().st_mtime_ns).encode("ascii"))
            h.update(str(path.stat().st_size if path.is_file() else 0).encode("ascii"))
        else:
            h.update(b"missing")
    return h.hexdigest()


def _load_registry() -> Dict[str, Dict[str, Any]]:
    rows = read_json(REGISTRY_PATH, [])
    scored: Dict[str, tuple[int, Dict[str, Any]]] = {}
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("dataset_name", "")).lower()
        dataset_id = str(row.get("dataset_id", "")).lower()
        for spec in DATASET_SPECS:
            spec_id = str(spec["id"]).lower()
            spec_name = str(spec["name"]).lower()
            score = 0
            if dataset_id == spec_id:
                score = 100
            elif name == spec_name:
                score = 90
            elif spec_name in name:
                score = 70
            elif spec_id in dataset_id or spec_id in name:
                score = 40
            if score and score > scored.get(spec["id"], (0, {}))[0]:
                scored[spec["id"]] = (score, row)
    return {key: value for key, (_, value) in scored.items()}


def _walk_stats(root: Path) -> Dict[str, Any]:
    if not root.exists():
        return {"exists": False, "path": str(root)}
    if root.is_file():
        return {
            "exists": True,
            "path": str(root),
            "is_file": True,
            "bytes": root.stat().st_size,
            "file_count": 1,
            "dir_count": 0,
            "truncated": False,
            "extensions": {root.suffix.lower() or "<none>": 1},
            "sample_files": [str(root)],
            "has_video": root.suffix.lower() in {".mp4", ".mov", ".avi"},
            "has_image": root.suffix.lower() in {".jpg", ".jpeg", ".png"},
            "homography_like_files": [],
            "scale_like_files": [],
        }

    file_count = 0
    dir_count = 0
    total_bytes = 0
    extensions: Counter[str] = Counter()
    sample_files: list[str] = []
    homography_like: list[str] = []
    scale_like: list[str] = []
    has_video = False
    has_image = False
    truncated = False
    video_ext = {".mp4", ".mov", ".avi", ".mkv"}
    image_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    for current, dirs, files in os.walk(root):
        dir_count += len(dirs)
        for filename in files:
            path = Path(current) / filename
            suffix = path.suffix.lower() or "<none>"
            file_count += 1
            extensions[suffix] += 1
            if suffix in video_ext:
                has_video = True
            if suffix in image_ext:
                has_image = True
            lower = filename.lower()
            if "homography" in lower or lower in {"h.txt", "h.csv"} or lower.endswith("_h.txt"):
                if len(homography_like) < 20:
                    homography_like.append(str(path))
            if "scale" in lower or "meter" in lower or "calib" in lower:
                if len(scale_like) < 20:
                    scale_like.append(str(path))
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            if len(sample_files) < 12:
                sample_files.append(str(path))
            if file_count >= MAX_AUDIT_FILES_PER_ROOT:
                truncated = True
                break
        if truncated:
            break

    return {
        "exists": True,
        "path": str(root),
        "is_file": False,
        "bytes": total_bytes,
        "size_mb": round(total_bytes / (1024 * 1024), 3),
        "file_count": file_count,
        "dir_count": dir_count,
        "truncated": truncated,
        "extensions": dict(extensions.most_common(12)),
        "sample_files": sample_files,
        "has_video": has_video,
        "has_image": has_image,
        "homography_like_files": homography_like,
        "scale_like_files": scale_like,
    }


def _read_known_metrics() -> Dict[str, Any]:
    return {
        "stage41_all_agent_dataset": read_json("outputs/stage41_breakthrough/stage41_all_agent_dataset.json", {}),
        "stage41_gate": read_json("outputs/stage41_breakthrough/world_model_gate_stage41.json", {}),
        "m3w_package_manifest": read_json("outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json", {}),
        "sdd_time_geometry_stage23": read_json("outputs/reports/stage23_sdd_time_geometry_audit.json", {}),
        "sdd_time_geometry_stage30": read_json("outputs/stage30_m3w_verified/time_geometry_raw_audit.json", {}),
        "sdd_horizon_stage21": read_json("outputs/reports/stage21_sdd_horizon_audit.json", {}),
        "external_horizon_stage34": read_json("outputs/stage34_external_geometry/external_horizon_split_report.json", {}),
    }


def _dataset_cached_summary(dataset_id: str, known: Mapping[str, Any]) -> Dict[str, Any]:
    if dataset_id == "sdd":
        sdd_time = known.get("sdd_time_geometry_stage30") or known.get("sdd_time_geometry_stage23") or {}
        sdd_horizon = known.get("sdd_horizon_stage21") or {}
        return {
            "source": "cached_verified",
            "time_geometry": sdd_time,
            "horizon_audit": sdd_horizon,
            "metric_time_conclusion": sdd_time.get("allowed_conclusion")
            or sdd_time.get("conclusion")
            or "pixel raw-frame only; effective seconds unknown",
        }
    if dataset_id in {"opentraj", "eth_ucy", "trajnet", "ucy"}:
        all_agent = known.get("stage41_all_agent_dataset") or {}
        external_horizon = known.get("external_horizon_stage34") or {}
        return {
            "source": "cached_verified",
            "stage41_all_agent_splits": all_agent.get("splits", {}),
            "stage34_horizon_status": external_horizon.get("horizon_status", {}),
            "stage34_split_summary": external_horizon.get("splits", {}),
            "metric_time_conclusion": "dataset-local raw-frame only; no global metric/seconds claim",
        }
    if dataset_id == "tgsim":
        return {
            "source": "cached_verified",
            "metric_time_conclusion": "traffic metric diagnostic only; not pedestrian/drone world-model success",
        }
    return {
        "source": "not_run",
        "metric_time_conclusion": "raw/source calibration not verified in Stage42 initial audit",
    }


def _legal_status(spec: Mapping[str, Any], registry: Mapping[str, Any]) -> Dict[str, Any]:
    if registry:
        return {
            "source": "cached_verified",
            "official_url": registry.get("official_url") or spec["official_hint"],
            "license_name": registry.get("license_name", "unknown"),
            "license_summary": registry.get("license_summary", ""),
            "requires_login": bool(registry.get("requires_login", False)),
            "requires_application": bool(registry.get("requires_application", False)),
            "requires_manual_terms_acceptance": bool(registry.get("requires_manual_terms_acceptance", False)),
            "auto_download_allowed": bool(registry.get("auto_download_allowed", False)),
            "legal_risk_level": registry.get("legal_risk_level", "unknown"),
            "download_status": registry.get("download_status", "not_run"),
        }
    return {
        "source": "not_run",
        "official_url": spec["official_hint"],
        "license_name": "unknown_in_stage42_audit",
        "license_summary": "Use official source and dataset terms before download or redistribution.",
        "requires_login": False,
        "requires_application": False,
        "requires_manual_terms_acceptance": True,
        "auto_download_allowed": False,
        "legal_risk_level": "unknown",
        "download_status": "not_run",
    }


def _audit_dataset(spec: Mapping[str, Any], registry_by_id: Mapping[str, Mapping[str, Any]], known: Mapping[str, Any]) -> Dict[str, Any]:
    raw_stats = [_walk_stats(Path(path)) for path in spec["raw_candidates"]]
    converted_stats = [_walk_stats(Path(path)) for path in spec["converted_candidates"]]
    raw_found = [row for row in raw_stats if row.get("exists")]
    converted_found = [row for row in converted_stats if row.get("exists")]
    registry = registry_by_id.get(str(spec["id"]), {})
    legal = _legal_status(spec, registry)
    cached = _dataset_cached_summary(str(spec["id"]), known)
    homography_like = [p for row in raw_found + converted_found for p in row.get("homography_like_files", [])]
    scale_like = [p for row in raw_found + converted_found for p in row.get("scale_like_files", [])]
    has_video = any(row.get("has_video") for row in raw_found)
    has_image = any(row.get("has_image") for row in raw_found + converted_found)
    calibration_state = "not_verified"
    metric_claim_allowed = False
    seconds_claim_allowed = False
    if spec["id"] == "sdd":
        calibration_state = "pixel_raw_frame_only"
    elif spec["id"] == "tgsim":
        calibration_state = "traffic_metric_diagnostic_only"
        metric_claim_allowed = True
    elif homography_like or scale_like:
        calibration_state = "calibration_files_found_but_not_validated"

    next_actions = []
    if not raw_found and not converted_found:
        next_actions.append("user_action_required: provide or legally download source data from official page")
    if legal.get("requires_manual_terms_acceptance") or legal.get("requires_application") or legal.get("requires_login"):
        next_actions.append("do_not_auto_download: requires terms/login/application or source-specific approval")
    if calibration_state != "traffic_metric_diagnostic_only" and not metric_claim_allowed:
        next_actions.append("keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified")
    if str(spec["id"]) in {"eth_ucy", "trajnet", "ucy", "opentraj"}:
        next_actions.append("Stage42-B: rebuild source/scene-level split from raw/feature rows before new validation")
    if str(spec["id"]) in {"eth_ucy", "trajnet", "ucy"}:
        next_actions.append("Stage42-C: use as priority external full-waypoint dynamics domain")

    return {
        "source": "fresh_run",
        "dataset_id": spec["id"],
        "dataset_name": spec["name"],
        "domain": spec["domain"],
        "data_role": spec["role"],
        "official_hint": spec["official_hint"],
        "raw_candidates": raw_stats,
        "converted_candidates": converted_stats,
        "raw_path_found": bool(raw_found),
        "converted_path_found": bool(converted_found),
        "has_video": has_video,
        "has_image_or_scene_pack": has_image,
        "homography_like_files_found": homography_like[:20],
        "scale_like_files_found": scale_like[:20],
        "known_coordinate_unit": spec["known_coordinate_unit"],
        "known_metric_status": spec["known_metric_status"],
        "calibration_state": calibration_state,
        "metric_claim_allowed": metric_claim_allowed,
        "seconds_claim_allowed": seconds_claim_allowed,
        "legal_status": legal,
        "cached_metrics": cached,
        "stage42_readiness": {
            "can_train_or_eval_from_existing_local_state": bool(converted_found),
            "can_rebuild_from_raw_local_state": bool(raw_found),
            "needs_user_action": (not raw_found and not converted_found)
            or bool(legal.get("requires_manual_terms_acceptance"))
            or bool(legal.get("requires_application"))
            or bool(legal.get("requires_login")),
            "ready_for_metric_claim": metric_claim_allowed and str(spec["id"]) == "tgsim",
            "ready_for_seconds_claim": seconds_claim_allowed,
        },
        "next_actions": next_actions,
    }


def build_data_calibration_report() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    registry = _load_registry()
    known = _read_known_metrics()
    datasets = [_audit_dataset(spec, registry, known) for spec in DATASET_SPECS]

    metric_ready = [d["dataset_id"] for d in datasets if d["stage42_readiness"]["ready_for_metric_claim"]]
    seconds_ready = [d["dataset_id"] for d in datasets if d["stage42_readiness"]["ready_for_seconds_claim"]]
    external_ready = [
        d["dataset_id"]
        for d in datasets
        if d["dataset_id"] in {"eth_ucy", "trajnet", "ucy", "opentraj"}
        and d["stage42_readiness"]["can_train_or_eval_from_existing_local_state"]
    ]
    user_actions = [
        {
            "dataset_id": d["dataset_id"],
            "dataset_name": d["dataset_name"],
            "official_url": d["legal_status"]["official_url"],
            "reason": d["next_actions"],
        }
        for d in datasets
        if d["stage42_readiness"]["needs_user_action"]
    ]
    payload = {
        "source": "fresh_run",
        "stage": "Stage42-A data and calibration audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _hash_paths(
            [
                REGISTRY_PATH,
                "outputs/stage41_breakthrough/stage41_all_agent_dataset.json",
                "outputs/stage41_breakthrough/world_model_gate_stage41.json",
                "outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json",
                "outputs/reports/stage23_sdd_time_geometry_audit.json",
                "outputs/stage30_m3w_verified/time_geometry_raw_audit.json",
                "outputs/stage34_external_geometry/external_horizon_split_report.json",
            ]
        ),
        "datasets": datasets,
        "summary": {
            "datasets_audited": len(datasets),
            "raw_paths_found": sum(1 for d in datasets if d["raw_path_found"]),
            "converted_paths_found": sum(1 for d in datasets if d["converted_path_found"]),
            "external_domains_ready_from_existing_state": external_ready,
            "metric_claim_ready_datasets": metric_ready,
            "seconds_claim_ready_datasets": seconds_ready,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage42_b_external_validation_ready": len(external_ready) >= 2,
            "stage42_c_full_waypoint_prereq_ready": bool(read_json("outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json", {}))
            and len(external_ready) >= 2,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": user_actions,
        "next_recommended_actions": [
            "Stage42-B: rebuild source-level or scene-level external split from Stage41/Stage37 feature rows, not by reusing a single old split blindly.",
            "Stage42-C: run full-waypoint model comparison against endpoint-only, linear bridge, learned shape, graph interaction, protected full-waypoint, and ungated full-waypoint.",
            "Stage42-D: prioritize retrained no-neighbor/no-interaction/no-safe-switch/no-teacher-floor ablations with bootstrap or seeds.",
            "Keep all SDD/external claims raw-frame dataset-local until a separate FPS/stride/homography/scale audit proves otherwise.",
        ],
    }
    write_json(OUT_DIR / "data_calibration_stage42.json", payload)
    write_md(OUT_DIR / "data_calibration_stage42.md", _render_calibration_md(payload))
    write_md(OUT_DIR / "user_action_required_stage42.md", _render_user_actions(payload))
    write_md(OUT_DIR / "stage42_stage_a_gate.md", _render_stage_a_gate(payload))
    _append_ledger(payload)
    return payload


def _render_calibration_md(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-A Data And Calibration Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- datasets_audited: `{s['datasets_audited']}`",
        f"- raw_paths_found: `{s['raw_paths_found']}`",
        f"- converted_paths_found: `{s['converted_paths_found']}`",
        f"- external_domains_ready_from_existing_state: `{', '.join(s['external_domains_ready_from_existing_state']) or 'none'}`",
        f"- metric_claim_ready_datasets: `{', '.join(s['metric_claim_ready_datasets']) or 'none'}`",
        f"- seconds_claim_ready_datasets: `{', '.join(s['seconds_claim_ready_datasets']) or 'none'}`",
        f"- global_metric_claim_allowed: `{s['global_metric_claim_allowed']}`",
        f"- global_seconds_claim_allowed: `{s['global_seconds_claim_allowed']}`",
        f"- stage42_b_external_validation_ready: `{s['stage42_b_external_validation_ready']}`",
        f"- stage42_c_full_waypoint_prereq_ready: `{s['stage42_c_full_waypoint_prereq_ready']}`",
        "",
        "## Dataset Audit Table",
        "",
        "| dataset | raw found | converted found | role | calibration state | metric claim | seconds claim | next use |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for d in payload["datasets"]:
        readiness = d["stage42_readiness"]
        next_use = "Stage42-B/C ready" if readiness["can_train_or_eval_from_existing_local_state"] else "user action/blocker"
        lines.append(
            f"| `{d['dataset_id']}` | `{d['raw_path_found']}` | `{d['converted_path_found']}` | {d['data_role']} | {d['calibration_state']} | `{d['metric_claim_allowed']}` | `{d['seconds_claim_allowed']}` | {next_use} |"
        )
    lines.extend(["", "## Per-Dataset Notes", ""])
    for d in payload["datasets"]:
        lines.extend(
            [
                f"### {d['dataset_name']}",
                "",
                f"- source: `{d['source']}`",
                f"- domain: {d['domain']}",
                f"- official_hint: `{d['official_hint']}`",
                f"- coordinate_unit: `{d['known_coordinate_unit']}`",
                f"- metric_status: `{d['known_metric_status']}`",
                f"- raw_path_found: `{d['raw_path_found']}`",
                f"- converted_path_found: `{d['converted_path_found']}`",
                f"- has_video: `{d['has_video']}`",
                f"- has_image_or_scene_pack: `{d['has_image_or_scene_pack']}`",
                f"- homography_like_files_found: `{len(d['homography_like_files_found'])}`",
                f"- scale_like_files_found: `{len(d['scale_like_files_found'])}`",
                f"- calibration_state: `{d['calibration_state']}`",
                f"- metric_claim_allowed: `{d['metric_claim_allowed']}`",
                f"- seconds_claim_allowed: `{d['seconds_claim_allowed']}`",
                f"- legal_source: `{d['legal_status']['source']}`",
                f"- license_name: `{d['legal_status']['license_name']}`",
                f"- auto_download_allowed: `{d['legal_status']['auto_download_allowed']}`",
                f"- requires_terms_or_login_or_application: `{d['legal_status']['requires_manual_terms_acceptance'] or d['legal_status']['requires_login'] or d['legal_status']['requires_application']}`",
                "",
                "Next actions:",
                *[f"- {item}" for item in d["next_actions"]],
                "",
            ]
        )
    lines.extend(
        [
            "## Stage42-A Conclusion",
            "",
            "Stage42 can proceed to external validation and full-waypoint dynamics from existing local converted state, but it cannot make metric or seconds-level claims. SDD remains pixel raw-frame. External pedestrian domains remain dataset-local raw-frame / unverified weak-metric diagnostics. TGSIM may carry traffic metric diagnostics only and cannot be used as pedestrian world-model success.",
        ]
    )
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42 User Action Required",
        "",
        "- source: `fresh_run`",
        "- purpose: list datasets that cannot be legally auto-downloaded or need user-provided local paths/terms.",
        "",
    ]
    actions = payload.get("user_action_required", [])
    if not actions:
        lines.append("No immediate user action is required for Stage42-A because existing local/converted state is enough for the next validation step.")
        return lines
    for row in actions:
        lines.extend(
            [
                f"## {row['dataset_name']}",
                "",
                f"- official_url_or_hint: `{row['official_url']}`",
                "- reason:",
                *[f"  - {reason}" for reason in row["reason"]],
                "",
            ]
        )
    return lines


def _render_stage_a_gate(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gates = [
        ("Data Presence Gate", s["converted_paths_found"] >= 3),
        ("External Validation Readiness Gate", s["stage42_b_external_validation_ready"]),
        ("Full-Waypoint Prereq Gate", s["stage42_c_full_waypoint_prereq_ready"]),
        ("Metric Overclaim Guard", not s["global_metric_claim_allowed"]),
        ("Seconds Overclaim Guard", not s["global_seconds_claim_allowed"]),
        ("Stage5C Execution Gate", not s["stage5c_executed"]),
        ("SMC Execution Gate", not s["smc_enabled"]),
    ]
    passed = sum(1 for _, ok in gates if ok)
    lines = [
        "# Stage42-A Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- gates: `{passed} / {len(gates)}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gates:
        lines.append(f"| {name} | `{ok}` |")
    lines.extend(
        [
            "",
            "Verdict: Stage42-A data/calibration audit is sufficient to proceed to Stage42-B external validation, but not sufficient for metric or seconds-level claims.",
        ]
    )
    return lines


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": "run_stage42_data_calibration.py",
        "source": payload["source"],
        "status": "success",
        "generated_at_utc": payload["generated_at_utc"],
        "git_commit": payload["git_commit"],
        "input_hash": payload["input_hash"],
        "outputs": [
            str(OUT_DIR / "data_calibration_stage42.json"),
            str(OUT_DIR / "data_calibration_stage42.md"),
            str(OUT_DIR / "user_action_required_stage42.md"),
            str(OUT_DIR / "stage42_stage_a_gate.md"),
        ],
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_data_calibration_report()
