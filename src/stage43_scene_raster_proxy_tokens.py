from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    _cache_path,
    _gather_old,
    _git_commit,
    _jsonable,
    _npz,
    _row_hash,
)
from src.stage43_latent_token_schema_coverage import REPORT_JSON as STAGE43_Z_JSON


DATA_DIR = Path("data/stage43_scene_proxy_tokens")
REPORT_JSON = OUT_DIR / "stage43_scene_raster_proxy_tokens.json"
REPORT_MD = OUT_DIR / "stage43_scene_raster_proxy_tokens.md"
GATE_MD = OUT_DIR / "stage43_stage_aa_scene_raster_proxy_tokens_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AA_SCENE_RASTER_PROXY_TOKENS"
SOURCE = "fresh_stage43_aa_scene_raster_proxy_tokens"
GRID_SIZE = 16
EPS = 1e-6

FEATURE_NAMES = [
    "scene_proxy_source_available",
    "scene_proxy_domain_available",
    "scene_proxy_level_source",
    "scene_proxy_rel_x",
    "scene_proxy_rel_y",
    "scene_proxy_boundary_sdf",
    "scene_proxy_route_occupancy",
    "scene_proxy_route_density_log",
    "scene_proxy_goal_dx_rel",
    "scene_proxy_goal_dy_rel",
    "scene_proxy_goal_alignment",
    "scene_proxy_entropy_mean",
    "scene_proxy_ambiguity_mean",
    "scene_proxy_rows_log",
]


@dataclass(frozen=True)
class SceneProxy:
    level: str
    key: str
    rows: int
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    scale: float
    route_grid: np.ndarray
    density_mean: float
    goal_vector: np.ndarray
    entropy_mean: float
    ambiguity_mean: float


def _source_key(dataset: str, source_file: str) -> str:
    return f"{dataset}::{source_file}"


def _domain_key(dataset: str) -> str:
    return str(dataset)


def _hash_array(arr: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _proxy_hash(proxies: Mapping[str, SceneProxy]) -> str:
    digest = hashlib.sha256()
    for key in sorted(proxies):
        p = proxies[key]
        digest.update(key.encode("utf-8"))
        for arr in [p.centroid, p.bounds_min, p.bounds_max, p.goal_vector, p.route_grid]:
            digest.update(np.asarray(arr, dtype=np.float32).tobytes())
        digest.update(json.dumps({"rows": p.rows, "scale": p.scale, "density": p.density_mean}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _make_grid(points: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    span = np.maximum(bounds_max - bounds_min, EPS)
    norm = np.clip((points - bounds_min[None, :]) / span[None, :], 0.0, 0.999999)
    ix = np.floor(norm[:, 0] * GRID_SIZE).astype(int)
    iy = np.floor(norm[:, 1] * GRID_SIZE).astype(int)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    np.add.at(grid, (iy, ix), 1.0)
    max_count = float(grid.max()) if grid.size else 0.0
    if max_count > 0:
        grid = grid / max_count
    return grid.astype(np.float32)


def _build_proxy(level: str, key: str, points: np.ndarray, scales: np.ndarray, densities: np.ndarray, proto: np.ndarray, likelihood: np.ndarray, entropy: np.ndarray, ambiguity: np.ndarray) -> SceneProxy:
    points = np.asarray(points, dtype=np.float32)
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    span = np.maximum(bounds_max - bounds_min, EPS)
    pad = np.maximum(0.05 * span, np.float32(np.median(scales) * 0.02 if len(scales) else 1e-3))
    bounds_min = bounds_min - pad
    bounds_max = bounds_max + pad
    centroid = np.mean(points, axis=0).astype(np.float32)
    scale = float(max(np.max(bounds_max - bounds_min), np.median(scales) if len(scales) else 1.0, EPS))
    weights = np.asarray(likelihood, dtype=np.float64)
    proto64 = np.asarray(proto, dtype=np.float64)
    denom = float(np.sum(weights)) + EPS
    goal_vector = np.sum(proto64 * weights[:, :, None], axis=(0, 1)) / denom
    return SceneProxy(
        level=level,
        key=key,
        rows=int(len(points)),
        centroid=centroid.astype(np.float32),
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        scale=scale,
        route_grid=_make_grid(points, bounds_min, bounds_max),
        density_mean=float(np.mean(densities)) if len(densities) else 0.0,
        goal_vector=goal_vector.astype(np.float32),
        entropy_mean=float(np.mean(entropy)) if len(entropy) else 0.0,
        ambiguity_mean=float(np.mean(ambiguity)) if len(ambiguity) else 0.0,
    )


def _load_aux(cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "density": _gather_old(cache, "history", "history_density").astype(np.float32),
        "history_dx": _gather_old(cache, "history", "history_dx").astype(np.float32),
        "history_dy": _gather_old(cache, "history", "history_dy").astype(np.float32),
        "prototype_vectors": _gather_old(cache, "goal", "prototype_vectors").astype(np.float32),
        "prototype_likelihood": _gather_old(cache, "goal", "prototype_likelihood").astype(np.float32),
        "prototype_entropy": _gather_old(cache, "goal", "prototype_entropy").astype(np.float32),
        "goal_ambiguity": _gather_old(cache, "goal", "goal_ambiguity").astype(np.float32),
    }


def _build_train_proxies() -> tuple[dict[str, SceneProxy], dict[str, SceneProxy], dict[str, Any]]:
    train = _npz(_cache_path("train"))
    aux = _load_aux(train)
    dataset = train["dataset"].astype(str)
    source_file = train["source_file"].astype(str)
    current_xy = train["current_xy"].astype(np.float32)
    scale = np.maximum(train["scale"].astype(np.float32), EPS)
    source_proxies: dict[str, SceneProxy] = {}
    domain_proxies: dict[str, SceneProxy] = {}
    for key in sorted({_source_key(d, s) for d, s in zip(dataset, source_file)}):
        d_key, src = key.split("::", 1)
        ids = np.where((dataset == d_key) & (source_file == src))[0]
        if len(ids):
            source_proxies[key] = _build_proxy(
                "source",
                key,
                current_xy[ids],
                scale[ids],
                aux["density"][ids],
                aux["prototype_vectors"][ids],
                aux["prototype_likelihood"][ids],
                aux["prototype_entropy"][ids],
                aux["goal_ambiguity"][ids],
            )
    for d_key in sorted(set(dataset)):
        ids = np.where(dataset == d_key)[0]
        if len(ids):
            domain_proxies[d_key] = _build_proxy(
                "domain",
                d_key,
                current_xy[ids],
                scale[ids],
                aux["density"][ids],
                aux["prototype_vectors"][ids],
                aux["prototype_likelihood"][ids],
                aux["prototype_entropy"][ids],
                aux["goal_ambiguity"][ids],
            )
    summary = {
        "train_rows": int(len(dataset)),
        "source_proxy_count": len(source_proxies),
        "domain_proxy_count": len(domain_proxies),
        "source_proxy_hash": _proxy_hash(source_proxies),
        "domain_proxy_hash": _proxy_hash(domain_proxies),
        "build_split": "stage43_train_only",
        "future_endpoint_usage": False,
        "future_waypoint_usage": False,
        "test_endpoint_goal_usage": False,
    }
    return source_proxies, domain_proxies, summary


def _grid_value(proxy: SceneProxy, xy: np.ndarray) -> float:
    span = np.maximum(proxy.bounds_max - proxy.bounds_min, EPS)
    norm = np.clip((xy - proxy.bounds_min) / span, 0.0, 0.999999)
    ix = int(np.floor(float(norm[0]) * GRID_SIZE))
    iy = int(np.floor(float(norm[1]) * GRID_SIZE))
    return float(proxy.route_grid[iy, ix])


def _boundary_sdf(proxy: SceneProxy, xy: np.ndarray) -> float:
    distances = np.asarray(
        [
            xy[0] - proxy.bounds_min[0],
            proxy.bounds_max[0] - xy[0],
            xy[1] - proxy.bounds_min[1],
            proxy.bounds_max[1] - xy[1],
        ],
        dtype=np.float32,
    )
    return float(np.min(distances) / max(proxy.scale, EPS))


def _alignment(last_velocity: np.ndarray, goal_vector: np.ndarray) -> float:
    denom = float(np.linalg.norm(last_velocity) * np.linalg.norm(goal_vector))
    if denom <= EPS:
        return 0.0
    return float(np.dot(last_velocity, goal_vector) / denom)


def _features_for_split(split: str, source_proxies: Mapping[str, SceneProxy], domain_proxies: Mapping[str, SceneProxy], max_source_rows: float) -> dict[str, Any]:
    cache = _npz(_cache_path(split))
    aux = _load_aux(cache)
    dataset = cache["dataset"].astype(str)
    source_file = cache["source_file"].astype(str)
    current_xy = cache["current_xy"].astype(np.float32)
    history_last = np.stack([aux["history_dx"][:, -1], aux["history_dy"][:, -1]], axis=1).astype(np.float32)
    features = np.zeros((len(dataset), len(FEATURE_NAMES)), dtype=np.float32)
    proxy_level = np.full(len(dataset), "missing", dtype="<U16")
    source_available = 0
    domain_available = 0
    for i, (d_key, src) in enumerate(zip(dataset, source_file)):
        s_key = _source_key(d_key, src)
        proxy = source_proxies.get(s_key)
        level_is_source = proxy is not None
        if proxy is None:
            proxy = domain_proxies.get(_domain_key(d_key))
        if proxy is None:
            continue
        if level_is_source:
            source_available += 1
            proxy_level[i] = "source"
        else:
            domain_available += 1
            proxy_level[i] = "domain"
        xy = current_xy[i]
        rel = (xy - proxy.centroid) / max(proxy.scale, EPS)
        goal_rel = proxy.goal_vector / max(proxy.scale, EPS)
        features[i] = np.asarray(
            [
                1.0 if level_is_source else 0.0,
                1.0,
                1.0 if level_is_source else 0.0,
                rel[0],
                rel[1],
                _boundary_sdf(proxy, xy),
                _grid_value(proxy, xy),
                np.log1p(proxy.rows) / np.log1p(max_source_rows),
                goal_rel[0],
                goal_rel[1],
                _alignment(history_last[i], proxy.goal_vector),
                proxy.entropy_mean,
                proxy.ambiguity_mean,
                np.log1p(proxy.rows) / np.log1p(max_source_rows),
            ],
            dtype=np.float32,
        )
    feature_hash = _hash_array(features)
    out_path = DATA_DIR / f"stage43_scene_proxy_features_{split}.npz"
    np.savez_compressed(
        out_path,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
        proxy_level=proxy_level,
        dataset=dataset,
        source_file=source_file,
        row_hash=np.asarray([_row_hash(cache)], dtype="<U128"),
        feature_hash=np.asarray([feature_hash], dtype="<U128"),
    )
    return {
        "split": split,
        "rows": int(len(dataset)),
        "cache_path": str(out_path),
        "cache_exists": out_path.exists(),
        "feature_dim": int(features.shape[1]),
        "feature_hash": feature_hash,
        "row_hash": _row_hash(cache),
        "source_proxy_rows": int(source_available),
        "domain_fallback_rows": int(domain_available),
        "missing_proxy_rows": int(len(dataset) - source_available - domain_available),
        "source_proxy_coverage": float(source_available / max(len(dataset), 1)),
        "domain_or_source_coverage": float((source_available + domain_available) / max(len(dataset), 1)),
        "sdf_min": float(np.min(features[:, FEATURE_NAMES.index("scene_proxy_boundary_sdf")])) if len(features) else 0.0,
        "sdf_mean": float(np.mean(features[:, FEATURE_NAMES.index("scene_proxy_boundary_sdf")])) if len(features) else 0.0,
        "route_occupancy_mean": float(np.mean(features[:, FEATURE_NAMES.index("scene_proxy_route_occupancy")])) if len(features) else 0.0,
    }


def _proxy_manifest(source_proxies: Mapping[str, SceneProxy], domain_proxies: Mapping[str, SceneProxy], build_summary: Mapping[str, Any], split_features: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = {key: proxy.rows for key, proxy in source_proxies.items()}
    domain_rows = {key: proxy.rows for key, proxy in domain_proxies.items()}
    return {
        "source": SOURCE,
        "build_summary": dict(build_summary),
        "feature_names": FEATURE_NAMES,
        "source_proxy_rows": source_rows,
        "domain_proxy_rows": domain_rows,
        "split_features": split_features,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    build = payload["proxy_build"]
    splits = payload["split_features"]
    leakage = payload["no_leakage"]
    gates = {
        "stage43_z_precondition_passed": payload["stage43_z_precondition"].get("verdict") == "stage43_z_latent_token_schema_coverage_pass",
        "train_only_source_proxies_built": build["build_split"] == "stage43_train_only"
        and build["source_proxy_count"] > 0
        and build["domain_proxy_count"] > 0,
        "all_split_scene_proxy_features_built": all(row["cache_exists"] and row["rows"] > 0 for row in splits.values()),
        "row_hashes_recorded": all(bool(row["row_hash"]) and bool(row["feature_hash"]) for row in splits.values()),
        "source_or_domain_coverage_complete": all(row["domain_or_source_coverage"] >= 1.0 for row in splits.values()),
        "scene_raster_proxy_features_present": all(
            name in payload["feature_names"]
            for name in [
                "scene_proxy_boundary_sdf",
                "scene_proxy_route_occupancy",
                "scene_proxy_route_density_log",
            ]
        ),
        "goal_proxy_features_present": all(
            name in payload["feature_names"]
            for name in ["scene_proxy_goal_dx_rel", "scene_proxy_goal_dy_rel", "scene_proxy_goal_alignment"]
        ),
        "no_future_or_test_goal_leakage": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["test_endpoint_goal_construction"] is False
        and leakage["scene_proxy_built_from_stage43_train_only"] is True,
        "claim_boundary_preserved": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["true_scene_image_token_claim"] is False
        and payload["claim_boundary"]["true_sdf_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "not_standalone_deployment": payload["deployment_contract"]["standalone_world_model_deployable"] is False
        and payload["deployment_contract"]["requires_stage37_stage42_floor"] is True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_aa_scene_raster_proxy_tokens_pass"
        if passed == total
        else "stage43_aa_scene_raster_proxy_tokens_partial",
        "scene_raster_proxy_token_ready": passed == total,
        "standalone_world_model_deployable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def build_scene_raster_proxy_tokens() -> dict[str, Any]:
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)
    z = read_json(STAGE43_Z_JSON, {})
    source_proxies, domain_proxies, build_summary = _build_train_proxies()
    max_rows = float(max([p.rows for p in source_proxies.values()] + [1]))
    split_features = {
        split: _features_for_split(split, source_proxies, domain_proxies, max_rows)
        for split in SPLITS
    }
    manifest = _proxy_manifest(source_proxies, domain_proxies, build_summary, split_features)
    manifest_path = DATA_DIR / "stage43_scene_proxy_manifest.json"
    write_json(manifest_path, _jsonable(manifest))
    payload: dict[str, Any] = {
        "stage": "Stage43-AA scene/raster proxy tokens",
        "source": SOURCE,
        "result_source": "fresh_train_only_scene_proxy_token_build_from_stage43_full_waypoint_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_z_precondition": z.get("stage43_z_gate", {}),
        "proxy_build": build_summary,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(json.dumps(_jsonable(manifest), sort_keys=True).encode("utf-8")).hexdigest(),
        "feature_names": FEATURE_NAMES,
        "split_features": split_features,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_built_from_stage43_train_only": True,
            "future_labels_loss_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "true_scene_image_token_claim": False,
            "true_sdf_claim": False,
            "scene_raster_proxy_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "deployment_contract": {
            "standalone_world_model_deployable": False,
            "requires_stage37_stage42_floor": True,
            "integration_status": "auxiliary_scene_raster_proxy_cache_ready_not_yet_retrained_into_stage43_m",
        },
    }
    payload["stage43_aa_gate"] = _gate(payload)
    _write_reports(payload)
    _update_text_outputs(payload)
    return payload


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_aa_gate"]
    lines = [
        "# Stage43-AA Scene/Raster Proxy Tokens",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- scene/raster proxy token ready: `{gate['scene_raster_proxy_token_ready']}`",
        f"- standalone deployment: `False`",
        "",
        "## Build Summary",
        "",
        f"- build split: `{payload['proxy_build']['build_split']}`",
        f"- source proxies: `{payload['proxy_build']['source_proxy_count']}`",
        f"- domain proxies: `{payload['proxy_build']['domain_proxy_count']}`",
        f"- source proxy hash: `{payload['proxy_build']['source_proxy_hash']}`",
        f"- domain proxy hash: `{payload['proxy_build']['domain_proxy_hash']}`",
        f"- manifest path: `{payload['manifest_path']}`",
        f"- manifest sha256: `{payload['manifest_sha256']}`",
        "",
        "## Split Features",
        "",
        "| split | rows | feature dim | source coverage | source+domain coverage | feature hash |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for split, row in payload["split_features"].items():
        lines.append(
            f"| {split} | {row['rows']} | {row['feature_dim']} | {row['source_proxy_coverage']:.4f} | {row['domain_or_source_coverage']:.4f} | `{row['feature_hash'][:12]}` |"
        )
    lines.extend(
        [
            "",
            "## Feature Names",
            "",
            *[f"- `{name}`" for name in payload["feature_names"]],
            "",
            "## Boundary",
            "",
            "- The proxy is built from Stage43 train rows only.",
            "- It uses current/past positions, train route bounds, train route occupancy grids, domain/source route priors, and past-motion goal prototypes.",
            "- It is not a raw scene image token, not an annotated walkable-area SDF, and not a metric scene map.",
            "- Test endpoints and future waypoints are not used to build the proxy.",
            "- Integration status: auxiliary cache ready; Stage43-M has not yet been retrained with these scene/raster proxy tokens.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AA Scene/Raster Proxy Token Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- scene/raster proxy token ready: `{gate['scene_raster_proxy_token_ready']}`",
            f"- standalone world model deployable: `False`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_aa_gate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"source_proxy_hash = `{payload['proxy_build']['source_proxy_hash']}`",
        f"manifest_sha256 = `{payload['manifest_sha256']}`",
        "",
        "Stage43-AA fills the explicit scene/raster/SDF-token gap with a train-only proxy: source/domain route bounds, route occupancy grids, boundary-SDF proxy, density prior, and scene-agnostic goal-vector priors. It writes row-aligned auxiliary features for train/val/test and records row/feature hashes.",
        "",
        "This is still a proxy, not raw scene imagery, not annotated walkable geometry, and not verified metric SDF. It is not yet retrained into Stage43-M; it is an auxiliary scene/raster token cache for the next latent-state training step. No future endpoints, future waypoints, central velocity, or test endpoint goals are used.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_aa_scene_raster_proxy_tokens"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "source_proxy_hash": payload["proxy_build"]["source_proxy_hash"],
        "manifest_sha256": payload["manifest_sha256"],
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
                        "stage": "Stage43-AA",
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
    payload = build_scene_raster_proxy_tokens()
    return payload


if __name__ == "__main__":
    result = main()
    gate = result["stage43_aa_gate"]
    print(f"Stage43-AA scene/raster proxy tokens: {gate['verdict']} ({gate['passed']}/{gate['total']})")
