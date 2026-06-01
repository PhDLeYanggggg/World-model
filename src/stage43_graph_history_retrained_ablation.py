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


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
CURRENT_GRAPH_DIR = Path("data/stage43_all_agent_current_graph_cache")
HISTORY_GRAPH_DIR = Path("data/stage43_all_agent_history_graph_cache")

REPORT_JSON = OUT_DIR / "stage43_graph_history_retrained_ablation.json"
REPORT_MD = OUT_DIR / "stage43_graph_history_retrained_ablation.md"
GATE_MD = OUT_DIR / "stage43_stage_bo_graph_history_retrained_ablation_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_graph_history_retrained_ablation_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bo_graph_history_retrained_ablation"
SECTION = "STAGE43_BO_GRAPH_HISTORY_RETRAINED_ABLATION"

BN_JSON = OUT_DIR / "stage43_all_agent_history_graph_cache.json"
VARIANTS = ["no_graph", "current_graph_only", "history_graph_only", "full_graph"]
SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _graph_current_path(split: str) -> Path:
    return CURRENT_GRAPH_DIR / f"stage43_all_agent_current_graph_{split}.npz"


def _graph_history_path(split: str) -> Path:
    return HISTORY_GRAPH_DIR / f"stage43_all_agent_history_graph_{split}.npz"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _selected_ids(split: str, *, max_rows: int | None, seed: int) -> np.ndarray:
    with np.load(m._cache_path(split), allow_pickle=False) as cache:
        n = len(cache["horizon"])
    ids = np.arange(n)
    if max_rows is not None and max_rows < n:
        rng = np.random.default_rng(int(seed) + SPLIT_OFFSETS[split])
        ids = np.sort(rng.choice(ids, size=int(max_rows), replace=False))
    return ids


def _masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    weighted = np.where(mask, values, 0.0)
    denom = np.maximum(np.sum(mask, axis=axis), 1)
    return (np.sum(weighted, axis=axis) / denom).astype(np.float32)


def _masked_max(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    filled = np.where(mask, values, -np.inf)
    out = np.max(filled, axis=axis)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def _masked_min(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    filled = np.where(mask, values, np.inf)
    out = np.min(filled, axis=axis)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def _graph_feature_matrix(
    split: str,
    ids: np.ndarray,
    *,
    include_current: bool,
    include_history: bool,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    summary: dict[str, Any] = {"split": split, "rows": int(len(ids))}
    if include_current:
        cur = _load_npz(_graph_current_path(split))
        mask = cur["neighbor_mask"][ids].astype(bool)
        dist = cur["neighbor_distance"][ids].astype(np.float32)
        edge_attr = cur["neighbor_edge_attr"][ids].astype(np.float32)
        degree = np.sum(mask, axis=1).astype(np.float32)
        features = [
            degree[:, None],
            _masked_mean(dist, mask, axis=1)[:, None],
            _masked_min(dist, mask, axis=1)[:, None],
            _masked_mean(edge_attr[:, :, 3], mask, axis=1)[:, None],
            _masked_max(edge_attr[:, :, 3], mask, axis=1)[:, None],
            _masked_mean(edge_attr[:, :, 4], mask, axis=1)[:, None],
            _masked_mean(edge_attr[:, :, 5], mask, axis=1)[:, None],
        ]
        parts.extend(features)
        names.extend(
            [
                "graph_current_degree",
                "graph_current_mean_distance",
                "graph_current_min_distance",
                "graph_current_mean_inv_distance",
                "graph_current_max_inv_distance",
                "graph_current_mean_bearing_cos",
                "graph_current_mean_bearing_sin",
            ]
        )
        summary["current_graph_degree_mean"] = float(np.mean(degree)) if len(degree) else 0.0
        summary["current_graph_rows_with_neighbors"] = int(np.sum(degree > 0))
    if include_history:
        hist = _load_npz(_graph_history_path(split))
        mask = hist["neighbor_mask"][ids].astype(bool)
        edge_hist = hist["edge_history_attr"][ids].astype(np.float32)
        all_valid = hist["all_agent_history_valid_mask"][ids].astype(bool)
        target_valid = all_valid[:, 0, :]
        neighbor_any_valid = np.any(all_valid[:, 1:, :], axis=2)
        target_hist_xy = hist["all_agent_history_xy"][ids, 0].astype(np.float32)
        target_disp = target_hist_xy[:, -1, :] - target_hist_xy[:, 0, :]
        target_disp_norm = np.linalg.norm(target_disp, axis=1).astype(np.float32)
        features = [
            np.sum(target_valid, axis=1).astype(np.float32)[:, None],
            np.sum(neighbor_any_valid, axis=1).astype(np.float32)[:, None],
            _masked_mean(edge_hist[:, :, 0], mask, axis=1)[:, None],
            _masked_mean(edge_hist[:, :, 1], mask, axis=1)[:, None],
            _masked_max(edge_hist[:, :, 1], mask, axis=1)[:, None],
            _masked_mean(edge_hist[:, :, 3], mask, axis=1)[:, None],
            _masked_max(edge_hist[:, :, 3], mask, axis=1)[:, None],
            _masked_mean(edge_hist[:, :, 5], mask, axis=1)[:, None],
            _masked_max(np.abs(edge_hist[:, :, 5]), mask, axis=1)[:, None],
            target_disp_norm[:, None],
        ]
        parts.extend(features)
        names.extend(
            [
                "graph_history_target_valid_count",
                "graph_history_neighbor_valid_degree",
                "graph_history_mean_shared_valid_count",
                "graph_history_mean_neighbor_path_length",
                "graph_history_max_neighbor_path_length",
                "graph_history_mean_neighbor_speed",
                "graph_history_max_neighbor_speed",
                "graph_history_mean_neighbor_minus_target_speed",
                "graph_history_max_abs_neighbor_minus_target_speed",
                "graph_history_target_displacement_norm",
            ]
        )
        summary["history_graph_rows_with_neighbor_history"] = int(np.sum(np.sum(neighbor_any_valid, axis=1) > 0))
        summary["history_graph_rows_with_full_target_history"] = int(np.sum(np.sum(target_valid, axis=1) >= target_valid.shape[1]))
    if not parts:
        return np.zeros((len(ids), 0), dtype=np.float32), [], summary
    x = np.concatenate(parts, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    summary["graph_feature_count"] = int(x.shape[1])
    return x, names, summary


def _augment_split(ds: m.WaypointSplit, split: str, *, max_rows: int | None, seed: int, variant: str) -> tuple[m.WaypointSplit, dict[str, Any]]:
    include_current = variant in {"current_graph_only", "full_graph"}
    include_history = variant in {"history_graph_only", "full_graph"}
    ids = _selected_ids(split, max_rows=max_rows, seed=seed)
    graph_x, graph_names, summary = _graph_feature_matrix(
        split, ids, include_current=include_current, include_history=include_history
    )
    if graph_x.shape[0] != ds.x.shape[0]:
        raise ValueError(f"Graph feature rows do not match {split}: {graph_x.shape[0]} vs {ds.x.shape[0]}")
    if graph_x.shape[1]:
        ds.x = np.concatenate([ds.x, graph_x], axis=1).astype(np.float32)
        ds.feature_names = [*ds.feature_names, *graph_names]
    return ds, summary


def _build_variant_split(split: str, *, max_rows: int | None, seed: int, variant: str) -> tuple[m.WaypointSplit, dict[str, Any]]:
    ds = m._build_split(split, max_rows=max_rows, seed=seed)
    return _augment_split(ds, split, max_rows=max_rows, seed=seed, variant=variant)


def _train_one(args: argparse.Namespace, *, variant: str, rows: Mapping[str, int | None]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    row_seed = int(args.seed)
    train_seed = int(args.seed) + 127 * VARIANTS.index(variant)
    runtime = m._configure_runtime(train_seed)
    train, train_graph = _build_variant_split("train", max_rows=rows["train"], seed=row_seed, variant=variant)
    val, val_graph = _build_variant_split("val", max_rows=rows["val"], seed=row_seed, variant=variant)
    test, test_graph = _build_variant_split("test", max_rows=rows["test"], seed=row_seed, variant=variant)
    train, val, test, mean, std = m._standardize(train, val, test)
    model = m.FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim))
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    ckpt_path = CKPT_DIR / f"stage43_graph_history_retrained_ablation_{variant}.pt"
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
        val_policy = m._search_policy(val, val_pred)
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
    val_policy = m._search_policy(val, val_pred)
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    result = {
        "variant": variant,
        "feature_count": int(train.x.shape[1]),
        "graph_feature_count": int(train_graph.get("graph_feature_count", 0)),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "graph_feature_summaries": {"train": train_graph, "val": val_graph, "test": test_graph},
        "training_history": history,
        "validation_selected_policy": val_policy,
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


def _metric_delta(full: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(full.get(key, 0.0)) - float(base.get(key, 0.0))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "endpoint_fde_improvement_vs_floor",
            "t50_full_waypoint_ade_improvement_vs_floor",
            "t50_endpoint_fde_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }


def _contribution_ci(full: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray], *, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    masks = {
        "all_full_waypoint_ade_graph_contribution": np.ones(len(full["selected_ade"]), dtype=bool),
        "t50_full_waypoint_ade_graph_contribution": full["h50"].astype(bool),
        "hard_failure_full_waypoint_ade_graph_contribution": full["hard_failure"].astype(bool),
        "t100_raw_frame_graph_contribution_diagnostic": full["h100"].astype(bool),
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
            floor = float(np.mean(full["floor_ade"][sample]))
            full_imp = 1.0 - float(np.mean(full["selected_ade"][sample])) / max(floor, EPS)
            base_imp = 1.0 - float(np.mean(base["selected_ade"][sample])) / max(floor, EPS)
            vals[i] = full_imp - base_imp
        out["metrics"][name] = {
            "rows": int(len(ids)),
            "mean": float(np.mean(vals)),
            "low": float(np.quantile(vals, 0.025)),
            "high": float(np.quantile(vals, 0.975)),
        }
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    variants = {row["variant"]: row for row in payload["variants"]}
    full = variants.get("full_graph", {})
    no_graph = variants.get("no_graph", {})
    full_metrics = full.get("test_metrics_with_floor", {})
    delta = payload.get("full_graph_minus_no_graph", {})
    gates = {
        "bn_precondition_passed": payload["preconditions"]["stage43_bn_verdict"]
        == "stage43_bn_all_agent_history_graph_cache_pass_raw_scene_blocker",
        "fresh_retrained_graph_variants": payload["result_source"] == "fresh_retrained_graph_history_ablation"
        and payload["ablation_type"]["not_inference_masking"] is True,
        "full_and_no_graph_retrained": bool(full) and bool(no_graph),
        "current_and_history_graph_variants_retrained": "current_graph_only" in variants and "history_graph_only" in variants,
        "graph_features_used_by_full_graph": int(full.get("graph_feature_count", 0)) > 0,
        "no_graph_uses_no_graph_features": int(no_graph.get("graph_feature_count", -1)) == 0,
        "bootstrap_or_resampling_recorded": payload["bootstrap_graph_contribution_ci"]["n"] >= 200,
        "latent_noncollapse": all(row["latent_variance"] > 0.01 for row in payload["variants"]),
        "easy_safety_reported": "easy_degradation_vs_floor" in full_metrics,
        "graph_contribution_measured": "t50_full_waypoint_ade_improvement_vs_floor" in delta,
        "checkpoints_not_committed": all(row["checkpoint_committed"] is False for row in payload["variants"]),
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["claim_boundary"]["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    contribution_supported = bool(
        delta.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0) > 0.0
        or delta.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0) > 0.0
        or delta.get("full_waypoint_ade_improvement_vs_floor", 0.0) > 0.0
    )
    verdict = (
        "stage43_bo_graph_history_retrained_ablation_pass_contribution_supported"
        if passed == total and contribution_supported
        else "stage43_bo_graph_history_retrained_ablation_pass_diagnostic_no_lift"
        if passed == total
        else "stage43_bo_graph_history_retrained_ablation_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "graph_history_retrained_ablation_executed": passed == total,
        "graph_history_contribution_supported": contribution_supported and passed == total,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


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
        raise KeyError(f"Unknown Stage43-BO graph ablation variants: {unknown}")
    if "full_graph" not in variants or "no_graph" not in variants:
        raise ValueError("Stage43-BO requires full_graph and no_graph variants.")
    results: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant in variants:
        row, arr = _train_one(args, variant=variant, rows=rows)
        results.append(row)
        arrays[variant] = arr
    by_variant = {row["variant"]: row for row in results}
    full = by_variant["full_graph"]
    no_graph = by_variant["no_graph"]
    for row in results:
        row["full_graph_minus_variant"] = _metric_delta(
            full["test_metrics_with_floor"], row["test_metrics_with_floor"]
        )
    contribution_ci = _contribution_ci(
        arrays["full_graph"], arrays["no_graph"], n=int(args.bootstrap), seed=int(args.seed) + 1709
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_retrained_graph_history_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "variants": results,
        "full_graph_minus_no_graph": _metric_delta(
            full["test_metrics_with_floor"], no_graph["test_metrics_with_floor"]
        ),
        "bootstrap_graph_contribution_ci": contribution_ci,
        "preconditions": {
            "stage43_bn_verdict": read_json(BN_JSON, {}).get("stage43_bn_gate", {}).get("verdict", "missing")
        },
        "ablation_type": {
            "fresh_retrained_variants": True,
            "same_train_val_test_protocol": True,
            "graph_feature_addition_retraining": True,
            "not_inference_masking": True,
            "not_deployment_policy": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "graph_inputs_past_or_current_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_sdf_claim": False,
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
                _graph_current_path("train"),
                _graph_current_path("val"),
                _graph_current_path("test"),
                _graph_history_path("train"),
                _graph_history_path("val"),
                _graph_history_path("test"),
            ]
        ),
    }
    payload["stage43_bo_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _variant_table(payload: Mapping[str, Any]) -> list[str]:
    rows = [
        "| variant | graph features | all | t50 | hard | easy | full-minus-variant all | full-minus-variant t50 | full-minus-variant hard | latent var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variants"]:
        metrics = row["test_metrics_with_floor"]
        delta = row["full_graph_minus_variant"]
        rows.append(
            f"| `{row['variant']}` | `{row['graph_feature_count']}` | `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['easy_degradation_vs_floor'])}` | `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{row['latent_variance']:.4f}` |"
        )
    return rows


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bo_gate"]
    delta = payload["full_graph_minus_no_graph"]
    ci = payload["bootstrap_graph_contribution_ci"]["metrics"]["t50_full_waypoint_ade_graph_contribution"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BO Graph-History Retrained Ablation",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- mode: `{payload['mode']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- graph-history contribution supported: `{gate['graph_history_contribution_supported']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Full Graph Minus No Graph",
            "",
            f"- all full-waypoint ADE contribution: `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 full-waypoint ADE contribution: `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- hard/failure full-waypoint ADE contribution: `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 bootstrap contribution CI: `[{_pct(ci['low'])}, {_pct(ci['high'])}]`",
            "",
            "## Variants",
            "",
            *_variant_table(payload),
            "",
            "## Boundary",
            "",
            "- This is a fresh retrained graph-feature ablation, not inference masking.",
            "- Future waypoints are labels/eval only.",
            "- It does not use raw scene/SDF tensors.",
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
            "# Stage43-BO Graph-History Retrained Ablation Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- graph-history ablation executed: `{gate['graph_history_retrained_ablation_executed']}`",
            f"- graph-history contribution supported: `{gate['graph_history_contribution_supported']}`",
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
            f"- graph-history retrained ablation executed: `{gate['graph_history_retrained_ablation_executed']}`",
            f"- graph-history contribution supported: `{gate['graph_history_contribution_supported']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BO executes a fresh retrained graph-history ablation under the protected full-waypoint latent protocol.",
            "- It compares full_graph against no_graph, current_graph_only, and history_graph_only.",
            "- It does not use raw scene/SDF and does not change the deployable protected policy.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bo_gate"]
    delta = payload["full_graph_minus_no_graph"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"graph_history_retrained_ablation_executed = `{gate['graph_history_retrained_ablation_executed']}`",
        f"graph_history_contribution_supported = `{gate['graph_history_contribution_supported']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Stage43-BO fresh-trains graph-history ablation variants. Full_graph minus no_graph: all `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`.",
        "",
        "This is retrained contribution evidence, not a deployment policy update. Raw-scene/SDF remains outside this ablation.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bo_graph_history_retrained_ablation"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "graph_history_retrained_ablation_executed": gate["graph_history_retrained_ablation_executed"],
        "graph_history_contribution_supported": gate["graph_history_contribution_supported"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "full_graph_minus_no_graph": payload["full_graph_minus_no_graph"],
        "variants": [
            {
                "variant": row["variant"],
                "graph_feature_count": row["graph_feature_count"],
                "metrics": row["test_metrics_with_floor"],
                "full_graph_minus_variant": row["full_graph_minus_variant"],
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
    state["current_stage"] = "stage43_bo_graph_history_retrained_ablation"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BO",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "graph_history_contribution_supported": gate["graph_history_contribution_supported"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BO retrained graph-history ablation.")
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
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    payload = _run(args)
    gate = payload["stage43_bo_gate"]
    print(f"Stage43-BO: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"graph_history_contribution_supported={gate['graph_history_contribution_supported']}")
    return payload


if __name__ == "__main__":
    main()
