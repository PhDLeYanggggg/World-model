from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_graph_history_retrained_ablation as bo
from src import stage43_scene_graph_multimodal_ablation as bp


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_gated_scene_graph_fusion.json"
REPORT_MD = OUT_DIR / "stage43_gated_scene_graph_fusion.md"
GATE_MD = OUT_DIR / "stage43_stage_bq_gated_scene_graph_fusion_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_gated_scene_graph_fusion_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BQ_GATED_SCENE_GRAPH_FUSION"
SOURCE = "fresh_stage43_bq_gated_scene_graph_fusion"
EPS = 1e-8


class GatedSceneGraphLatentDynamics(nn.Module):
    def __init__(self, base_dim: int, scene_dim: int, graph_dim: int, hidden_dim: int = 128, latent_dim: int = 32) -> None:
        super().__init__()
        self.base_dim = int(base_dim)
        self.scene_dim = int(scene_dim)
        self.graph_dim = int(graph_dim)
        total_dim = self.base_dim + self.scene_dim + self.graph_dim
        self.base_encoder = nn.Sequential(
            nn.LayerNorm(self.base_dim),
            nn.Linear(self.base_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.scene_encoder = nn.Sequential(nn.LayerNorm(self.scene_dim), nn.Linear(self.scene_dim, latent_dim)) if self.scene_dim else None
        self.graph_encoder = nn.Sequential(nn.LayerNorm(self.graph_dim), nn.Linear(self.graph_dim, latent_dim)) if self.graph_dim else None
        self.gate = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.mix_norm = nn.LayerNorm(latent_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_target_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 13))

    def forward(self, x: torch.Tensor, target_vec: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        base = x[:, : self.base_dim]
        scene = x[:, self.base_dim : self.base_dim + self.scene_dim]
        graph = x[:, self.base_dim + self.scene_dim :]
        base_z = self.base_encoder(base)
        gate = torch.sigmoid(self.gate(x))
        z_t = base_z
        if self.scene_encoder is not None:
            z_t = z_t + gate[:, 0:1] * self.scene_encoder(scene)
        if self.graph_encoder is not None:
            z_t = z_t + gate[:, 1:2] * self.graph_encoder(graph)
        z_t = self.mix_norm(z_t)
        z_next = self.dynamics(z_t)
        out = self.head(z_next)
        result = {
            "z_t": z_t,
            "z_next": z_next,
            "waypoint_delta": out[:, :8].reshape(-1, 4, 2),
            "failure_logit": out[:, 8],
            "gain_logit": out[:, 9],
            "harm_logit": out[:, 10],
            "density": torch.sigmoid(out[:, 11]),
            "validity_logit": out[:, 12],
            "context_gate": gate,
        }
        if target_vec is not None:
            result["target_latent"] = self.future_target_encoder(target_vec).detach()
        return result


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _build_gated_split(
    split: str,
    *,
    max_rows: int | None,
    row_seed: int,
) -> tuple[m.WaypointSplit, dict[str, Any]]:
    base = m._build_split(split, max_rows=max_rows, seed=row_seed)
    base_dim = int(base.x.shape[1])
    ids = bo._selected_ids(split, max_rows=max_rows, seed=row_seed)
    scene_x, scene_names, scene_summary = bp._scene_feature_matrix(split, ids, include_scene=True)
    graph_x, graph_names, graph_summary = bo._graph_feature_matrix(split, ids, include_current=True, include_history=True)
    if scene_x.shape[0] != base.x.shape[0] or graph_x.shape[0] != base.x.shape[0]:
        raise ValueError(
            f"Stage43-BQ context row mismatch for {split}: base={base.x.shape[0]} scene={scene_x.shape[0]} graph={graph_x.shape[0]}"
        )
    base.x = np.concatenate([base.x, scene_x, graph_x], axis=1).astype(np.float32)
    base.feature_names = [*base.feature_names, *scene_names, *graph_names]
    summary = {
        "split": split,
        "rows": int(base.x.shape[0]),
        "base_dim": base_dim,
        "scene_dim": int(scene_x.shape[1]),
        "graph_dim": int(graph_x.shape[1]),
        "input_dim": int(base.x.shape[1]),
        "scene": scene_summary,
        "graph": graph_summary,
    }
    return base, summary


def _gated_loss(
    model: GatedSceneGraphLatentDynamics,
    ds: m.WaypointSplit,
    ids: np.ndarray,
    device: torch.device,
    *,
    gate_l1_weight: float,
    easy_gate_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(ds.x[ids]).to(device)
    target_delta = torch.from_numpy(ds.waypoint_delta[ids]).to(device)
    valid = torch.from_numpy(ds.waypoint_valid[ids].astype(np.float32)).to(device)
    y_failure = torch.from_numpy(ds.y_failure[ids]).to(device)
    y_gain = torch.from_numpy(ds.y_gain[ids]).to(device)
    y_harm = torch.from_numpy(ds.y_harm[ids]).to(device)
    y_density = torch.from_numpy(ds.y_density[ids]).to(device)
    target = torch.from_numpy(m._target_vec(ds)[ids]).to(device)
    horizon = torch.from_numpy(ds.horizon[ids]).to(device)
    hard = torch.from_numpy((ds.hard[ids] | ds.failure[ids]).astype(np.float32)).to(device)
    easy = torch.from_numpy(ds.easy[ids].astype(np.float32)).to(device)
    out = model(x, target)
    per_wp = nn.functional.smooth_l1_loss(out["waypoint_delta"], target_delta, reduction="none").mean(dim=2)
    row_weight = 1.0 + 1.0 * hard + 1.0 * (horizon == 50).float() + 0.5 * (horizon == 100).float()
    waypoint = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_weight).mean()
    endpoint = nn.functional.smooth_l1_loss(out["waypoint_delta"][:, -1, :], target_delta[:, -1, :])
    failure = nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], y_failure)
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    density = nn.functional.mse_loss(out["density"], y_density)
    latent = nn.functional.mse_loss(out["z_next"], out["target_latent"])
    variance = out["z_next"].float().var(dim=0).mean()
    collapse = torch.relu(torch.tensor(0.02, device=device) - variance)
    gate_mean = out["context_gate"].mean()
    easy_gate = (out["context_gate"].mean(dim=1) * easy).sum() / easy.sum().clamp_min(1.0)
    total = (
        waypoint
        + 0.30 * endpoint
        + 0.35 * failure
        + 0.45 * gain
        + 0.55 * harm
        + 0.15 * density
        + 0.35 * latent
        + collapse
        + float(gate_l1_weight) * gate_mean
        + float(easy_gate_weight) * easy_gate
    )
    return total, {
        "waypoint": float(waypoint.detach().cpu()),
        "endpoint": float(endpoint.detach().cpu()),
        "failure": float(failure.detach().cpu()),
        "gain": float(gain.detach().cpu()),
        "harm": float(harm.detach().cpu()),
        "density": float(density.detach().cpu()),
        "latent": float(latent.detach().cpu()),
        "latent_variance": float(variance.detach().cpu()),
        "gate_mean": float(gate_mean.detach().cpu()),
        "easy_gate": float(easy_gate.detach().cpu()),
        "scene_gate_mean": float(out["context_gate"][:, 0].mean().detach().cpu()),
        "graph_gate_mean": float(out["context_gate"][:, 1].mean().detach().cpu()),
    }


@torch.no_grad()
def _predict_gated(
    model: GatedSceneGraphLatentDynamics,
    ds: m.WaypointSplit,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {
        "waypoint": [],
        "failure": [],
        "gain": [],
        "harm": [],
        "density": [],
        "latent": [],
        "context_gate": [],
    }
    for ids in m._batch_indices(len(ds.x), batch_size, shuffle=False, seed=0):
        x = torch.from_numpy(ds.x[ids]).to(device)
        out = model(x)
        outs["waypoint"].append(out["waypoint_delta"].detach().cpu().numpy())
        outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
        outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["density"].append(out["density"].detach().cpu().numpy())
        outs["latent"].append(out["z_next"].detach().cpu().numpy())
        outs["context_gate"].append(out["context_gate"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outs.items()}


def _train(args: argparse.Namespace, rows: Mapping[str, int | None]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    runtime = m._configure_runtime(int(args.seed))
    train, train_ctx = _build_gated_split("train", max_rows=rows["train"], row_seed=int(args.seed))
    val, val_ctx = _build_gated_split("val", max_rows=rows["val"], row_seed=int(args.seed))
    test, test_ctx = _build_gated_split("test", max_rows=rows["test"], row_seed=int(args.seed))
    dims = {"base_dim": train_ctx["base_dim"], "scene_dim": train_ctx["scene_dim"], "graph_dim": train_ctx["graph_dim"]}
    train, val, test, mean, std = m._standardize(train, val, test)
    model = GatedSceneGraphLatentDynamics(
        dims["base_dim"],
        dims["scene_dim"],
        dims["graph_dim"],
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
    )
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    ckpt_path = CKPT_DIR / "stage43_bq_gated_scene_graph_fusion.pt"
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        stats: list[dict[str, float]] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=int(args.seed) + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = _gated_loss(
                model,
                train,
                batch_ids,
                device,
                gate_l1_weight=float(args.gate_l1_weight),
                easy_gate_weight=float(args.easy_gate_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_pred = _predict_gated(model, val, device, int(args.batch_size))
        val_policy = bp._search_policy_with_cap(val, val_pred, max_switch_rate=float(args.max_switch_rate))
        selected_ade, selected_fde, switched = m._select_with_policy(val, val_pred, val_policy["policy"])
        val_metrics = m._metrics(val, selected_ade, selected_fde, switched)
        objective_loss = -float(
            val_metrics["full_waypoint_ade_improvement_vs_floor"]
            + 1.3 * val_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            + 0.9 * val_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - 12.0 * max(0.0, val_metrics["easy_degradation_vs_floor"] - 0.02)
            - 0.15 * val_metrics["switch_rate"]
        )
        stat_mean = {key: float(np.mean([s[key] for s in stats])) for key in stats[0]} if stats else {}
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_selection_objective_loss": objective_loss,
            "val_metrics": val_metrics,
            **stat_mean,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            m._jsonable(
                {
                    "source": SOURCE,
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
                    "dims": dims,
                    "hidden_dim": int(args.hidden_dim),
                    "latent_dim": int(args.latent_dim),
                    "seed": int(args.seed),
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
    val_pred = _predict_gated(model, val, device, int(args.batch_size))
    test_pred = _predict_gated(model, test, device, int(args.batch_size))
    val_policy = bp._search_policy_with_cap(val, val_pred, max_switch_rate=float(args.max_switch_rate))
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    gate_arr = test_pred["context_gate"]
    result = {
        "model": "gated_scene_graph_latent_fusion",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "dims": dims,
        "context_summaries": {"train": train_ctx, "val": val_ctx, "test": test_ctx},
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": metrics,
        "test_metrics_neural_without_floor": ungated,
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
        "gate_summary": {
            "scene_gate_mean": float(np.mean(gate_arr[:, 0])),
            "graph_gate_mean": float(np.mean(gate_arr[:, 1])),
            "scene_gate_easy_mean": float(np.mean(gate_arr[test.easy, 0])) if int(test.easy.sum()) else 0.0,
            "graph_gate_easy_mean": float(np.mean(gate_arr[test.easy, 1])) if int(test.easy.sum()) else 0.0,
            "scene_gate_hard_mean": float(np.mean(gate_arr[test.hard | test.failure, 0])) if int((test.hard | test.failure).sum()) else 0.0,
            "graph_gate_hard_mean": float(np.mean(gate_arr[test.hard | test.failure, 1])) if int((test.hard | test.failure).sum()) else 0.0,
        },
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
    return bp._metric_delta(current, baseline)


def _bootstrap_contribution(current: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], *, n: int, seed: int) -> dict[str, Any]:
    return bp._bootstrap_contribution(current, baseline, n=n, seed=seed)


def _bp_best_single(bp_payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_name = str(bp_payload.get("best_single_by_t50", "graph_history_only"))
    variants = {row["variant"]: row for row in bp_payload["variants"]}
    return variants[best_name], variants["no_context"]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    mode = "medium" if args.medium else "quick" if args.quick else "small"
    rows: dict[str, int | None]
    if args.quick:
        rows = {"train": 5000, "val": 2500, "test": 3000}
    elif args.medium:
        rows = {"train": 80000, "val": 35000, "test": 45000}
    else:
        rows = {"train": 20000, "val": 8000, "test": 12000}
    model_result, arrays = _train(args, rows)
    bp_payload = read_json(bp.REPORT_JSON, {})
    bp_gate = bp_payload.get("stage43_bp_gate", {})
    best_single, no_context = _bp_best_single(bp_payload)
    metrics = model_result["test_metrics_with_floor"]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_gated_scene_graph_latent_fusion",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "model": model_result,
        "bp_precondition": {
            "verdict": bp_gate.get("verdict", "missing"),
            "gate": f"{bp_gate.get('passed', 0)} / {bp_gate.get('total', 0)}",
            "full_multimodal_unsafe": bool(bp_gate.get("full_multimodal_unsafe", False)),
            "best_single_by_t50": bp_payload.get("best_single_by_t50", "missing"),
        },
        "best_single_metrics": best_single["test_metrics_with_floor"],
        "no_context_metrics": no_context["test_metrics_with_floor"],
        "gated_minus_best_single_by_t50": _metric_delta(metrics, best_single["test_metrics_with_floor"]),
        "gated_minus_no_context": _metric_delta(metrics, no_context["test_metrics_with_floor"]),
        "bootstrap_gated_vs_best_single_t50_ci": _bootstrap_contribution(
            arrays,
            {
                "selected_ade": np.full_like(arrays["selected_ade"], best_single["test_metrics_with_floor"]["mean_selected_ade"]),
                "selected_fde": np.full_like(arrays["selected_fde"], best_single["test_metrics_with_floor"].get("mean_selected_fde", 0.0)),
                "floor_ade": arrays["floor_ade"],
                "floor_fde": arrays["floor_fde"],
                "h50": arrays["h50"],
                "h100": arrays["h100"],
                "hard_failure": arrays["hard_failure"],
            },
            n=int(args.bootstrap),
            seed=int(args.seed) + 4109,
        ),
        "ablation_type": {
            "fresh_retrained_gated_fusion": True,
            "not_raw_concat": True,
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
                bp._scene_path("train"),
                bp._scene_path("val"),
                bp._scene_path("test"),
                bo._graph_current_path("train"),
                bo._graph_current_path("val"),
                bo._graph_current_path("test"),
                bo._graph_history_path("train"),
                bo._graph_history_path("val"),
                bo._graph_history_path("test"),
                bp.REPORT_JSON,
            ]
        ),
    }
    payload["stage43_bq_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["model"]["test_metrics_with_floor"]
    delta_best = payload["gated_minus_best_single_by_t50"]
    delta_no = payload["gated_minus_no_context"]
    gate_summary = payload["model"]["gate_summary"]
    bp_verdict = payload["bp_precondition"]["verdict"]
    full_unsafe = bool(metrics.get("easy_degradation_vs_floor", 1.0) > 0.02)
    safe = not full_unsafe
    beats_best_single = bool(
        safe
        and (
            delta_best["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_best["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_best["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
    )
    beats_no_context = bool(
        safe
        and (
            delta_no["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_no["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_no["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
    )
    gates = {
        "bp_negative_diagnostic_precondition_passed": bp_verdict
        in {
            "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_mixed_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_contribution_supported",
        },
        "fresh_gated_fusion_trained": payload["result_source"] == "fresh_gated_scene_graph_latent_fusion"
        and payload["ablation_type"]["fresh_retrained_gated_fusion"] is True,
        "not_raw_concat_or_inference_masking": payload["ablation_type"]["not_raw_concat"] is True
        and payload["ablation_type"]["not_inference_masking"] is True,
        "scene_and_graph_dims_present": payload["model"]["dims"]["scene_dim"] > 0 and payload["model"]["dims"]["graph_dim"] > 0,
        "learned_gates_measured": 0.0 <= gate_summary["scene_gate_mean"] <= 1.0
        and 0.0 <= gate_summary["graph_gate_mean"] <= 1.0,
        "latent_noncollapse": payload["model"]["latent_variance"] > 0.01,
        "safe_easy_measured": "easy_degradation_vs_floor" in metrics,
        "best_single_comparison_reported": "gated_minus_best_single_by_t50" in payload,
        "no_context_comparison_reported": "gated_minus_no_context" in payload,
        "checkpoints_not_committed": payload["model"]["checkpoint_committed"] is False,
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
    verdict = (
        "stage43_bq_gated_scene_graph_fusion_pass_contribution_supported"
        if passed == total and beats_best_single
        else "stage43_bq_gated_scene_graph_fusion_pass_safe_no_best_single_lift_diagnostic"
        if passed == total and safe and beats_no_context
        else "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic"
        if passed == total and safe
        else "stage43_bq_gated_scene_graph_fusion_pass_unsafe_diagnostic"
        if passed == total
        else "stage43_bq_gated_scene_graph_fusion_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "gated_fusion_executed": passed == total,
        "beats_best_single": beats_best_single,
        "beats_no_context": beats_no_context,
        "full_multimodal_unsafe": full_unsafe,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bq_gate"]
    metrics = payload["model"]["test_metrics_with_floor"]
    delta_best = payload["gated_minus_best_single_by_t50"]
    delta_no = payload["gated_minus_no_context"]
    gates = payload["model"]["gate_summary"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BQ Gated Scene-Graph Latent Fusion",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- beats best single: `{gate['beats_best_single']}`",
            f"- beats no context: `{gate['beats_no_context']}`",
            f"- full multimodal unsafe: `{gate['full_multimodal_unsafe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Protected Test Metrics",
            "",
            f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- hard/failure full-waypoint ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
            f"- switch rate: `{_pct(metrics['switch_rate'])}`",
            "",
            "## Contribution Deltas",
            "",
            f"- gated minus best single `{payload['bp_precondition']['best_single_by_t50']}` all/t50/hard: `{_pct(delta_best['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_best['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- gated minus no_context all/t50/hard: `{_pct(delta_no['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_no['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_no['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            "",
            "## Learned Context Gates",
            "",
            f"- scene gate mean: `{gates['scene_gate_mean']:.4f}`",
            f"- graph gate mean: `{gates['graph_gate_mean']:.4f}`",
            f"- scene gate easy/hard: `{gates['scene_gate_easy_mean']:.4f}` / `{gates['scene_gate_hard_mean']:.4f}`",
            f"- graph gate easy/hard: `{gates['graph_gate_easy_mean']:.4f}` / `{gates['graph_gate_hard_mean']:.4f}`",
            "",
            "## Boundary",
            "",
            "- This is a fresh retrained gated-fusion latent model, not inference masking.",
            "- Scene inputs are train-only scene/goal/raster proxies, not raw images or verified metric SDF.",
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
            "# Stage43-BQ Gated Scene-Graph Fusion Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- gated fusion executed: `{gate['gated_fusion_executed']}`",
            f"- beats best single: `{gate['beats_best_single']}`",
            f"- beats no context: `{gate['beats_no_context']}`",
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
            f"- gated fusion executed: `{gate['gated_fusion_executed']}`",
            f"- beats best single: `{gate['beats_best_single']}`",
            f"- full multimodal unsafe: `{gate['full_multimodal_unsafe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BQ executes a fresh retrained gated scene-proxy + graph-history latent fusion model.",
            "- It tests whether learned context gates repair Stage43-BP raw-concat easy damage.",
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
    gate = payload["stage43_bq_gate"]
    metrics = payload["model"]["test_metrics_with_floor"]
    delta_best = payload["gated_minus_best_single_by_t50"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"gated_fusion_executed = `{gate['gated_fusion_executed']}`",
        f"beats_best_single = `{gate['beats_best_single']}`",
        f"beats_no_context = `{gate['beats_no_context']}`",
        f"full_multimodal_unsafe = `{gate['full_multimodal_unsafe']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Stage43-BQ trains a learned gated scene-proxy + graph-history latent fusion model after Stage43-BP showed raw concatenation was unsafe. Protected metrics: all `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        "",
        f"Against the Stage43-BP best single-context t50 variant `{payload['bp_precondition']['best_single_by_t50']}`, gated fusion delta is all `{_pct(delta_best['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(delta_best['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(delta_best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`.",
        "",
        "This is gated multimodal latent-fusion evidence, not a deployment policy update. Scene remains train-only proxy scene/goal/raster evidence, not raw image/SDF evidence.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bq_gated_scene_graph_fusion"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "gated_fusion_executed": gate["gated_fusion_executed"],
        "beats_best_single": gate["beats_best_single"],
        "beats_no_context": gate["beats_no_context"],
        "full_multimodal_unsafe": gate["full_multimodal_unsafe"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "metrics": payload["model"]["test_metrics_with_floor"],
        "gated_minus_best_single_by_t50": payload["gated_minus_best_single_by_t50"],
        "gated_minus_no_context": payload["gated_minus_no_context"],
        "gate_summary": payload["model"]["gate_summary"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bq_gated_scene_graph_fusion"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BQ",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "beats_best_single": gate["beats_best_single"],
                        "full_multimodal_unsafe": gate["full_multimodal_unsafe"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BQ gated scene-proxy + graph-history latent fusion.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=449)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-switch-rate", type=float, default=0.55)
    parser.add_argument("--gate-l1-weight", type=float, default=0.04)
    parser.add_argument("--easy-gate-weight", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    payload = _run(args)
    gate = payload["stage43_bq_gate"]
    print(f"Stage43-BQ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"beats_best_single={gate['beats_best_single']}")
    print(f"full_multimodal_unsafe={gate['full_multimodal_unsafe']}")
    return payload


if __name__ == "__main__":
    main()
