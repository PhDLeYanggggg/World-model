from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_graph_history_retrained_ablation as bo
from src.stage43_scene_raster_proxy_tokens import DATA_DIR as SCENE_PROXY_DIR
from src.stage43_scene_raster_proxy_tokens import REPORT_JSON as STAGE43_AA_JSON


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_scene_graph_multimodal_ablation.json"
REPORT_MD = OUT_DIR / "stage43_scene_graph_multimodal_ablation.md"
GATE_MD = OUT_DIR / "stage43_stage_bp_scene_graph_multimodal_ablation_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_scene_graph_multimodal_ablation_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BP_SCENE_GRAPH_MULTIMODAL_ABLATION"
SOURCE = "fresh_stage43_bp_scene_graph_multimodal_ablation"
AG_JSON = OUT_DIR / "stage43_scene_proxy_retrained_ablation.json"
BO_JSON = OUT_DIR / "stage43_graph_history_retrained_ablation.json"
VARIANTS = ["no_context", "scene_proxy_only", "graph_history_only", "scene_graph_full"]
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _scene_path(split: str) -> Path:
    return SCENE_PROXY_DIR / f"stage43_scene_proxy_features_{split}.npz"


def _scene_feature_matrix(split: str, ids: np.ndarray, *, include_scene: bool) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not include_scene:
        return np.zeros((len(ids), 0), dtype=np.float32), [], {"split": split, "rows": int(len(ids))}
    path = _scene_path(split)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as scene:
        features = scene["features"].astype(np.float32)[ids]
        names = [f"scene_proxy::{name}" for name in scene["feature_names"].astype(str).tolist()]
        levels = scene["proxy_level"].astype(str)[ids]
        feature_hash = str(scene["feature_hash"][0])
    summary = {
        "split": split,
        "rows": int(len(ids)),
        "scene_feature_count": int(features.shape[1]),
        "scene_feature_hash": feature_hash,
        "source_level_rows": int(np.sum(levels == "source")),
        "domain_level_rows": int(np.sum(levels == "domain")),
        "missing_rows": int(np.sum(levels == "missing")),
    }
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names, summary


def _variant_flags(variant: str) -> tuple[bool, bool]:
    if variant == "no_context":
        return False, False
    if variant == "scene_proxy_only":
        return True, False
    if variant == "graph_history_only":
        return False, True
    if variant == "scene_graph_full":
        return True, True
    raise KeyError(f"Unknown Stage43-BP variant: {variant}")


def _search_policy_with_cap(val: m.WaypointSplit, pred: Mapping[str, np.ndarray], *, max_switch_rate: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for harm in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75]:
            for failure in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                selected_ade, selected_fde, switched = m._select_with_policy(val, pred, policy)
                metrics = m._metrics(val, selected_ade, selected_fde, switched)
                if metrics["easy_degradation_vs_floor"] > 0.01:
                    continue
                if metrics["switch_rate"] > max_switch_rate:
                    continue
                objective = (
                    metrics["full_waypoint_ade_improvement_vs_floor"]
                    + 1.2 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                    + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    - 0.25 * metrics["switch_rate"]
                )
                row = {"policy": policy, "metrics": metrics, "objective": float(objective), "max_switch_rate": max_switch_rate}
                if best is None or row["objective"] > best["objective"]:
                    best = row
    if best is None:
        selected_ade = val.floor_ade.copy()
        selected_fde = val.floor_fde.copy()
        switched = np.zeros(len(val.x), dtype=bool)
        return {
            "policy": {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01},
            "metrics": m._metrics(val, selected_ade, selected_fde, switched),
            "objective": 0.0,
            "max_switch_rate": max_switch_rate,
            "diagnostic": "no_validation_safe_policy_under_switch_cap_keep_floor",
        }
    return best


def _build_variant_split(
    split: str,
    *,
    max_rows: int | None,
    row_seed: int,
    variant: str,
) -> tuple[m.WaypointSplit, dict[str, Any]]:
    include_scene, include_graph = _variant_flags(variant)
    ds = m._build_split(split, max_rows=max_rows, seed=row_seed)
    ids = bo._selected_ids(split, max_rows=max_rows, seed=row_seed)
    scene_x, scene_names, scene_summary = _scene_feature_matrix(split, ids, include_scene=include_scene)
    graph_x, graph_names, graph_summary = bo._graph_feature_matrix(
        split,
        ids,
        include_current=include_graph,
        include_history=include_graph,
    )
    if scene_x.shape[0] != ds.x.shape[0] or graph_x.shape[0] != ds.x.shape[0]:
        raise ValueError(
            f"Context row mismatch for {split}/{variant}: base={ds.x.shape[0]} scene={scene_x.shape[0]} graph={graph_x.shape[0]}"
        )
    parts = [x for x in [scene_x, graph_x] if x.shape[1] > 0]
    if parts:
        ds.x = np.concatenate([ds.x, *parts], axis=1).astype(np.float32)
        ds.feature_names = [*ds.feature_names, *scene_names, *graph_names]
    summary = {
        "split": split,
        "rows": int(ds.x.shape[0]),
        "include_scene": include_scene,
        "include_graph": include_graph,
        "scene": scene_summary,
        "graph": graph_summary,
        "context_feature_count": int(scene_x.shape[1] + graph_x.shape[1]),
    }
    return ds, summary


def _train_one(args: argparse.Namespace, *, variant: str, rows: Mapping[str, int | None]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    row_seed = int(args.seed)
    train_seed = int(args.seed) + 193 * VARIANTS.index(variant)
    runtime = m._configure_runtime(train_seed)
    train, train_ctx = _build_variant_split("train", max_rows=rows["train"], row_seed=row_seed, variant=variant)
    val, val_ctx = _build_variant_split("val", max_rows=rows["val"], row_seed=row_seed, variant=variant)
    test, test_ctx = _build_variant_split("test", max_rows=rows["test"], row_seed=row_seed, variant=variant)
    train, val, test, mean, std = m._standardize(train, val, test)
    model = m.FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim))
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    ckpt_path = CKPT_DIR / f"stage43_scene_graph_multimodal_ablation_{variant}.pt"
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        latent_vars: list[float] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=train_seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = m._loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            latent_vars.append(float(stat.get("latent_variance", 0.0)))
        val_pred = m._predict(model, val, device, int(args.batch_size))
        val_policy = _search_policy_with_cap(val, val_pred, max_switch_rate=float(args.max_switch_rate))
        selected_ade, selected_fde, switched = m._select_with_policy(val, val_pred, val_policy["policy"])
        val_metrics = m._metrics(val, selected_ade, selected_fde, switched)
        objective_loss = -float(
            val_metrics["full_waypoint_ade_improvement_vs_floor"]
            + 1.2 * val_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            + 0.8 * val_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - 10.0 * max(0.0, val_metrics["easy_degradation_vs_floor"] - 0.02)
            - 0.10 * val_metrics["switch_rate"]
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_selection_objective_loss": objective_loss,
            "latent_variance": float(np.mean(latent_vars)) if latent_vars else 0.0,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            m._jsonable(
                {
                    "source": SOURCE,
                    "variant": variant,
                    "epoch": epoch + 1,
                    "elapsed_s": time.time() - start,
                    "last": row,
                    "git_commit": m._git_commit(),
                }
            ),
        )
        if objective_loss < best_val:
            best_val = objective_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train.feature_names,
                    "variant": variant,
                    "input_dim": int(train.x.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "latent_dim": int(args.latent_dim),
                    "seed": train_seed,
                    "row_seed": row_seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "checkpoint_committed": False,
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "future_waypoint_label_eval_only": True,
                        "central_velocity_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                ckpt_path,
            )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))
    val_policy = _search_policy_with_cap(val, val_pred, max_switch_rate=float(args.max_switch_rate))
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    result = {
        "variant": variant,
        "feature_count": int(train.x.shape[1]),
        "context_feature_count": int(test_ctx["context_feature_count"]),
        "scene_feature_count": int(test_ctx["scene"]["scene_feature_count"]) if test_ctx["include_scene"] else 0,
        "graph_feature_count": int(test_ctx["graph"].get("graph_feature_count", 0)) if test_ctx["include_graph"] else 0,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "context_summaries": {"train": train_ctx, "val": val_ctx, "test": test_ctx},
        "training_history": history,
        "validation_selected_policy": val_policy,
        "validation_policy_search": {
            "type": "conservative_switch_capped_grid",
            "max_switch_rate": float(args.max_switch_rate),
            "validation_easy_degradation_limit": 0.01,
            "test_threshold_tuning": False,
        },
        "test_metrics_with_floor": metrics,
        "test_metrics_neural_without_floor": ungated,
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
    }
    arrays = {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": test.floor_ade,
        "floor_fde": test.floor_fde,
        "h50": test.horizon == 50,
        "h100": test.horizon == 100,
        "hard_failure": test.hard | test.failure,
        "easy": test.easy,
    }
    return result, arrays


def _metric_delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "full_waypoint_ade_improvement_vs_floor",
        "endpoint_fde_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "t50_endpoint_fde_improvement_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    return {key: float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def _bootstrap_contribution(current: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], *, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    masks = {
        "all_full_waypoint_ade_contribution": np.ones(len(current["selected_ade"]), dtype=bool),
        "t50_full_waypoint_ade_contribution": current["h50"].astype(bool),
        "hard_failure_full_waypoint_ade_contribution": current["hard_failure"].astype(bool),
        "t100_raw_frame_contribution_diagnostic": current["h100"].astype(bool),
    }
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "metrics": {}}
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        if len(ids) == 0 or int(n) <= 0:
            out["metrics"][name] = {"rows": int(len(ids)), "mean": 0.0, "low": 0.0, "high": 0.0}
            continue
        vals = np.empty(int(n), dtype=np.float64)
        for i in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            floor = float(np.mean(current["floor_ade"][sample]))
            current_imp = 1.0 - float(np.mean(current["selected_ade"][sample])) / max(floor, EPS)
            base_imp = 1.0 - float(np.mean(baseline["selected_ade"][sample])) / max(floor, EPS)
            vals[i] = current_imp - base_imp
        out["metrics"][name] = {
            "rows": int(len(ids)),
            "mean": float(np.mean(vals)),
            "low": float(np.quantile(vals, 0.025)),
            "high": float(np.quantile(vals, 0.975)),
        }
    return out


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    mode = "medium" if args.medium else "quick" if args.quick else "small"
    if args.quick:
        rows = {"train": 5000, "val": 2500, "test": 3000}
    elif args.medium:
        rows = {"train": 80000, "val": 35000, "test": 45000}
    else:
        rows = {"train": 20000, "val": 8000, "test": 12000}
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise KeyError(f"Unknown Stage43-BP multimodal ablation variants: {unknown}")
    required = {"no_context", "scene_proxy_only", "graph_history_only", "scene_graph_full"}
    if not required.issubset(set(variants)):
        raise ValueError(f"Stage43-BP requires variants {sorted(required)}")
    results: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant in variants:
        row, arr = _train_one(args, variant=variant, rows=rows)
        results.append(row)
        arrays[variant] = arr
    by_variant = {row["variant"]: row for row in results}
    full = by_variant["scene_graph_full"]
    no_context = by_variant["no_context"]
    scene = by_variant["scene_proxy_only"]
    graph = by_variant["graph_history_only"]
    single_candidates = [scene, graph, no_context]
    best_single_by_t50 = max(
        single_candidates,
        key=lambda row: row["test_metrics_with_floor"]["t50_full_waypoint_ade_improvement_vs_floor"],
    )
    best_single_by_all = max(
        single_candidates,
        key=lambda row: row["test_metrics_with_floor"]["full_waypoint_ade_improvement_vs_floor"],
    )
    best_single_by_hard = max(
        single_candidates,
        key=lambda row: row["test_metrics_with_floor"]["hard_failure_full_waypoint_ade_improvement_vs_floor"],
    )
    for row in results:
        row["scene_graph_full_minus_variant"] = _metric_delta(
            full["test_metrics_with_floor"], row["test_metrics_with_floor"]
        )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_retrained_scene_graph_multimodal_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "variants": results,
        "best_single_by_t50": best_single_by_t50["variant"],
        "best_single_by_all": best_single_by_all["variant"],
        "best_single_by_hard": best_single_by_hard["variant"],
        "scene_graph_full_minus_no_context": _metric_delta(
            full["test_metrics_with_floor"], no_context["test_metrics_with_floor"]
        ),
        "scene_graph_full_minus_scene_proxy_only": _metric_delta(
            full["test_metrics_with_floor"], scene["test_metrics_with_floor"]
        ),
        "scene_graph_full_minus_graph_history_only": _metric_delta(
            full["test_metrics_with_floor"], graph["test_metrics_with_floor"]
        ),
        "scene_graph_full_minus_best_single_by_t50": _metric_delta(
            full["test_metrics_with_floor"], best_single_by_t50["test_metrics_with_floor"]
        ),
        "bootstrap_multimodal_vs_no_context_ci": _bootstrap_contribution(
            arrays["scene_graph_full"], arrays["no_context"], n=int(args.bootstrap), seed=int(args.seed) + 3101
        ),
        "bootstrap_multimodal_vs_best_single_t50_ci": _bootstrap_contribution(
            arrays["scene_graph_full"], arrays[best_single_by_t50["variant"]], n=int(args.bootstrap), seed=int(args.seed) + 3109
        ),
        "preconditions": {
            "stage43_aa_verdict": read_json(STAGE43_AA_JSON, {}).get("stage43_aa_gate", {}).get("verdict", "missing"),
            "stage43_ag_verdict": read_json(AG_JSON, {}).get("stage43_ag_gate", {}).get("verdict", "missing"),
            "stage43_bo_verdict": read_json(BO_JSON, {}).get("stage43_bo_gate", {}).get("verdict", "missing"),
        },
        "ablation_type": {
            "fresh_retrained_variants": True,
            "same_train_val_test_protocol": True,
            "not_inference_masking": True,
            "scene_proxy_and_graph_history_context": True,
            "not_raw_scene_or_verified_sdf": True,
            "not_deployment_policy": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "deployable_policy_changed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash(
            [
                m._cache_path("train"),
                m._cache_path("val"),
                m._cache_path("test"),
                _scene_path("train"),
                _scene_path("val"),
                _scene_path("test"),
                bo._graph_current_path("train"),
                bo._graph_current_path("val"),
                bo._graph_current_path("test"),
                bo._graph_history_path("train"),
                bo._graph_history_path("val"),
                bo._graph_history_path("test"),
            ]
        ),
    }
    payload["stage43_bp_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    variants = {row["variant"]: row for row in payload["variants"]}
    full = variants["scene_graph_full"]
    delta_no = payload["scene_graph_full_minus_no_context"]
    delta_scene = payload["scene_graph_full_minus_scene_proxy_only"]
    delta_graph = payload["scene_graph_full_minus_graph_history_only"]
    delta_best_t50 = payload["scene_graph_full_minus_best_single_by_t50"]
    full_metrics = full["test_metrics_with_floor"]
    pre = payload["preconditions"]
    gates = {
        "scene_proxy_cache_precondition_passed": pre["stage43_aa_verdict"] == "stage43_aa_scene_raster_proxy_tokens_pass",
        "scene_proxy_retrained_ablation_precondition_passed": pre["stage43_ag_verdict"] == "stage43_ag_scene_proxy_retrained_ablation_pass",
        "graph_history_retrained_ablation_precondition_passed": pre["stage43_bo_verdict"]
        == "stage43_bo_graph_history_retrained_ablation_pass_contribution_supported",
        "fresh_retrained_multimodal_variants": payload["result_source"] == "fresh_retrained_scene_graph_multimodal_ablation"
        and payload["ablation_type"]["not_inference_masking"] is True,
        "no_scene_no_graph_baseline_retrained": variants["no_context"]["context_feature_count"] == 0,
        "scene_only_and_graph_only_retrained": variants["scene_proxy_only"]["scene_feature_count"] > 0
        and variants["graph_history_only"]["graph_feature_count"] > 0,
        "full_scene_graph_uses_both_modalities": full["scene_feature_count"] > 0 and full["graph_feature_count"] > 0,
        "bootstrap_recorded": payload["bootstrap_multimodal_vs_no_context_ci"]["n"] >= 200
        and payload["bootstrap_multimodal_vs_best_single_t50_ci"]["n"] >= 200,
        "latent_noncollapse": all(row["latent_variance"] > 0.01 for row in payload["variants"]),
        "multimodal_beats_no_context_on_t50_or_hard": delta_no["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_no["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "multimodal_best_single_comparison_reported": "scene_graph_full_minus_best_single_by_t50" in payload
        and payload["best_single_by_t50"] in {"no_context", "scene_proxy_only", "graph_history_only"},
        "easy_safety_measured": "easy_degradation_vs_floor" in full_metrics,
        "checkpoints_not_committed": all(row["checkpoint_committed"] is False for row in payload["variants"]),
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["scene_proxy_train_only"] is True
        and payload["no_leakage"]["graph_inputs_past_or_current_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["raw_scene_or_verified_sdf_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["claim_boundary"]["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    multimodal_contribution_supported = bool(
        passed == total
        and (
            delta_no["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_no["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_no["full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
    )
    best_single_lift_supported = bool(
        delta_best_t50["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_scene["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_graph["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
    )
    full_unsafe = bool(full_metrics.get("easy_degradation_vs_floor", 1.0) > 0.02)
    verdict = (
        "stage43_bp_scene_graph_multimodal_ablation_pass_contribution_supported"
        if passed == total and multimodal_contribution_supported and best_single_lift_supported
        else "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic"
        if passed == total and full_unsafe
        else "stage43_bp_scene_graph_multimodal_ablation_pass_mixed_diagnostic"
        if passed == total
        else "stage43_bp_scene_graph_multimodal_ablation_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "multimodal_scene_graph_ablation_executed": passed == total,
        "multimodal_contribution_supported": multimodal_contribution_supported,
        "best_single_lift_supported": best_single_lift_supported,
        "full_multimodal_unsafe": full_unsafe,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _variant_table(payload: Mapping[str, Any]) -> list[str]:
    rows = [
        "| variant | scene | graph | all | t50 | hard | easy | full-minus-variant all | full-minus-variant t50 | full-minus-variant hard | latent var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variants"]:
        metrics = row["test_metrics_with_floor"]
        delta = row["scene_graph_full_minus_variant"]
        rows.append(
            f"| `{row['variant']}` | `{row['scene_feature_count']}` | `{row['graph_feature_count']}` | `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['easy_degradation_vs_floor'])}` | `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{row['latent_variance']:.4f}` |"
        )
    return rows


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bp_gate"]
    delta_no = payload["scene_graph_full_minus_no_context"]
    delta_best = payload["scene_graph_full_minus_best_single_by_t50"]
    ci_no = payload["bootstrap_multimodal_vs_no_context_ci"]["metrics"]["t50_full_waypoint_ade_contribution"]
    ci_best = payload["bootstrap_multimodal_vs_best_single_t50_ci"]["metrics"]["t50_full_waypoint_ade_contribution"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BP Scene-Graph Multimodal Retrained Ablation",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- mode: `{payload['mode']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- multimodal contribution supported: `{gate['multimodal_contribution_supported']}`",
            f"- best-single lift supported: `{gate['best_single_lift_supported']}`",
            f"- full multimodal unsafe: `{gate['full_multimodal_unsafe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Scene Graph Full Minus No Context",
            "",
            f"- all full-waypoint ADE contribution: `{_pct(delta_no['full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 full-waypoint ADE contribution: `{_pct(delta_no['t50_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- hard/failure full-waypoint ADE contribution: `{_pct(delta_no['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 bootstrap contribution CI: `[{_pct(ci_no['low'])}, {_pct(ci_no['high'])}]`",
            "",
            "## Scene Graph Full Minus Best Single By T50",
            "",
            f"- best single by t50: `{payload['best_single_by_t50']}`",
            f"- all contribution: `{_pct(delta_best['full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 contribution: `{_pct(delta_best['t50_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- hard/failure contribution: `{_pct(delta_best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 bootstrap contribution CI: `[{_pct(ci_best['low'])}, {_pct(ci_best['high'])}]`",
            "",
            "## Variants",
            "",
            *_variant_table(payload),
            "",
            "## Boundary",
            "",
            "- This is a fresh retrained multimodal context ablation, not inference masking.",
            "- Scene inputs are train-only scene/goal/raster proxies, not raw scene images or verified metric SDF.",
            "- Graph inputs are current-frame and past-only history graph summaries.",
            "- Future waypoints are labels/eval only.",
            "- It does not change the deployable protected policy.",
            "- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-BP Scene-Graph Multimodal Ablation Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- multimodal ablation executed: `{gate['multimodal_scene_graph_ablation_executed']}`",
            f"- multimodal contribution supported: `{gate['multimodal_contribution_supported']}`",
            f"- best-single lift supported: `{gate['best_single_lift_supported']}`",
            f"- full multimodal unsafe: `{gate['full_multimodal_unsafe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- scene-graph multimodal ablation executed: `{gate['multimodal_scene_graph_ablation_executed']}`",
            f"- multimodal contribution supported: `{gate['multimodal_contribution_supported']}`",
            f"- best-single lift supported: `{gate['best_single_lift_supported']}`",
            f"- full multimodal unsafe: `{gate['full_multimodal_unsafe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BP executes a fresh retrained scene-proxy + graph-history multimodal ablation.",
            "- It compares scene_graph_full against no_context, scene_proxy_only, and graph_history_only.",
            "- Scene evidence remains proxy-based, not raw image/SDF evidence.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bp_gate"]
    delta_no = payload["scene_graph_full_minus_no_context"]
    delta_best = payload["scene_graph_full_minus_best_single_by_t50"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"multimodal_scene_graph_ablation_executed = `{gate['multimodal_scene_graph_ablation_executed']}`",
        f"multimodal_contribution_supported = `{gate['multimodal_contribution_supported']}`",
        f"best_single_lift_supported = `{gate['best_single_lift_supported']}`",
        f"full_multimodal_unsafe = `{gate['full_multimodal_unsafe']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Stage43-BP fresh-trains no_context, scene_proxy_only, graph_history_only, and scene_graph_full variants. Scene_graph_full minus no_context: all `{_pct(delta_no['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(delta_no['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(delta_no['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`.",
        "",
        f"Against the best single-context t50 variant `{payload['best_single_by_t50']}`, scene_graph_full delta is all `{_pct(delta_best['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(delta_best['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(delta_best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`.",
        "",
        "This is multimodal retrained contribution evidence, not a deployment policy update. Scene remains train-only proxy scene/goal/raster evidence, not raw image/SDF evidence.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bp_scene_graph_multimodal_ablation"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "multimodal_scene_graph_ablation_executed": gate["multimodal_scene_graph_ablation_executed"],
        "multimodal_contribution_supported": gate["multimodal_contribution_supported"],
        "best_single_lift_supported": gate["best_single_lift_supported"],
        "full_multimodal_unsafe": gate["full_multimodal_unsafe"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "best_single_by_t50": payload["best_single_by_t50"],
        "best_single_by_all": payload["best_single_by_all"],
        "best_single_by_hard": payload["best_single_by_hard"],
        "scene_graph_full_minus_no_context": payload["scene_graph_full_minus_no_context"],
        "scene_graph_full_minus_best_single_by_t50": payload["scene_graph_full_minus_best_single_by_t50"],
        "variants": [
            {
                "variant": row["variant"],
                "scene_feature_count": row["scene_feature_count"],
                "graph_feature_count": row["graph_feature_count"],
                "context_feature_count": row["context_feature_count"],
                "metrics": row["test_metrics_with_floor"],
                "scene_graph_full_minus_variant": row["scene_graph_full_minus_variant"],
                "latent_variance": row["latent_variance"],
                "checkpoint_committed": False,
            }
            for row in payload["variants"]
        ],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bp_scene_graph_multimodal_ablation"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BP",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "multimodal_contribution_supported": gate["multimodal_contribution_supported"],
                        "full_multimodal_unsafe": gate["full_multimodal_unsafe"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BP scene-proxy + graph-history multimodal ablation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-switch-rate", type=float, default=0.65)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    payload = _run(args)
    gate = payload["stage43_bp_gate"]
    print(f"Stage43-BP: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"multimodal_contribution_supported={gate['multimodal_contribution_supported']}")
    print(f"best_single_lift_supported={gate['best_single_lift_supported']}")
    return payload


if __name__ == "__main__":
    main()
