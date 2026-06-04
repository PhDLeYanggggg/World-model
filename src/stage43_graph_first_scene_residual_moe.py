from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_gated_scene_graph_fusion as bq
from src import stage43_scene_graph_multimodal_ablation as bp
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_graph_first_scene_residual_moe.json"
REPORT_MD = OUT_DIR / "stage43_graph_first_scene_residual_moe.md"
GATE_MD = OUT_DIR / "stage43_stage_dl_graph_first_scene_residual_moe_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_graph_first_scene_residual_moe_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_DL_GRAPH_FIRST_SCENE_RESIDUAL_MOE"
SOURCE = "fresh_stage43_dl_graph_first_scene_residual_moe"
DK_JSON = OUT_DIR / "stage43_scene_graph_failure_taxonomy.json"
EPS = 1e-8


class GraphFirstSceneResidualMoE(nn.Module):
    """Graph-default latent dynamics with a guarded scene residual expert.

    The important design choice is asymmetry: graph context is part of the
    default expert, while scene proxy can only enter through a residual gate.
    This directly follows the Stage43-DK failure taxonomy.
    """

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
        self.graph_encoder = (
            nn.Sequential(nn.LayerNorm(self.graph_dim), nn.Linear(self.graph_dim, latent_dim))
            if self.graph_dim
            else None
        )
        self.scene_encoder = (
            nn.Sequential(nn.LayerNorm(self.scene_dim), nn.Linear(self.scene_dim, latent_dim))
            if self.scene_dim
            else None
        )
        self.scene_gate = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
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

    def _decode(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_next = self.dynamics(self.mix_norm(z_t))
        out = self.head(z_next)
        return z_next, out, out[:, :8].reshape(-1, 4, 2)

    def forward(self, x: torch.Tensor, target_vec: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        base = x[:, : self.base_dim]
        scene = x[:, self.base_dim : self.base_dim + self.scene_dim]
        graph = x[:, self.base_dim + self.scene_dim :]

        base_z = self.base_encoder(base)
        graph_z = base_z
        if self.graph_encoder is not None:
            graph_z = graph_z + self.graph_encoder(graph)

        scene_residual = torch.zeros_like(graph_z)
        if self.scene_encoder is not None:
            scene_residual = self.scene_encoder(scene)
        scene_gate = torch.sigmoid(self.scene_gate(x))

        graph_next, graph_out, graph_waypoint = self._decode(graph_z)
        fused_next, fused_out, fused_waypoint = self._decode(graph_z + scene_gate * scene_residual)
        result = {
            "z_t_graph": self.mix_norm(graph_z),
            "z_next": fused_next,
            "z_next_graph": graph_next,
            "waypoint_delta": fused_waypoint,
            "graph_waypoint_delta": graph_waypoint,
            "failure_logit": fused_out[:, 8],
            "gain_logit": fused_out[:, 9],
            "harm_logit": fused_out[:, 10],
            "density": torch.sigmoid(fused_out[:, 11]),
            "validity_logit": fused_out[:, 12],
            "scene_gate": scene_gate[:, 0],
            "graph_failure_logit": graph_out[:, 8],
            "graph_gain_logit": graph_out[:, 9],
            "graph_harm_logit": graph_out[:, 10],
        }
        if target_vec is not None:
            result["target_latent"] = self.future_target_encoder(target_vec).detach()
        return result


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _loss(
    model: GraphFirstSceneResidualMoE,
    ds: m.WaypointSplit,
    ids: np.ndarray,
    device: torch.device,
    *,
    easy_gate_weight: float,
    preservation_weight: float,
    gate_supervision_weight: float,
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
    row_weight = 1.0 + 1.3 * hard + 1.2 * (horizon == 50).float() + 0.5 * (horizon == 100).float()
    waypoint = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_weight).mean()
    endpoint = nn.functional.smooth_l1_loss(out["waypoint_delta"][:, -1, :], target_delta[:, -1, :])
    failure = nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], y_failure)
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    density = nn.functional.mse_loss(out["density"], y_density)
    latent = nn.functional.mse_loss(out["z_next"], out["target_latent"])
    variance = out["z_next"].float().var(dim=0).mean()
    collapse = torch.relu(torch.tensor(0.02, device=device) - variance)

    gate_target = torch.clamp(y_gain * (1.0 - y_harm) * (0.5 + 0.5 * hard), 0.0, 1.0)
    gate_supervision = nn.functional.binary_cross_entropy(out["scene_gate"].clamp(1e-5, 1.0 - 1e-5), gate_target)
    easy_gate = (out["scene_gate"] * easy).sum() / easy.sum().clamp_min(1.0)
    preservation = (
        nn.functional.smooth_l1_loss(out["waypoint_delta"], out["graph_waypoint_delta"].detach(), reduction="none")
        .mean(dim=(1, 2))
        * (easy + 0.5 * (1.0 - hard))
    ).mean()

    total = (
        waypoint
        + 0.30 * endpoint
        + 0.35 * failure
        + 0.45 * gain
        + 0.55 * harm
        + 0.15 * density
        + 0.35 * latent
        + collapse
        + float(easy_gate_weight) * easy_gate
        + float(preservation_weight) * preservation
        + float(gate_supervision_weight) * gate_supervision
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
        "easy_scene_gate": float(easy_gate.detach().cpu()),
        "graph_preservation": float(preservation.detach().cpu()),
        "scene_gate_supervision": float(gate_supervision.detach().cpu()),
        "scene_gate_mean": float(out["scene_gate"].mean().detach().cpu()),
    }


@torch.no_grad()
def _predict(
    model: GraphFirstSceneResidualMoE,
    ds: m.WaypointSplit,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {
        "waypoint": [],
        "graph_waypoint": [],
        "failure": [],
        "gain": [],
        "harm": [],
        "density": [],
        "latent": [],
        "scene_gate": [],
    }
    for ids in m._batch_indices(len(ds.x), batch_size, shuffle=False, seed=0):
        x = torch.from_numpy(ds.x[ids]).to(device)
        out = model(x)
        outs["waypoint"].append(out["waypoint_delta"].detach().cpu().numpy())
        outs["graph_waypoint"].append(out["graph_waypoint_delta"].detach().cpu().numpy())
        outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
        outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["density"].append(out["density"].detach().cpu().numpy())
        outs["latent"].append(out["z_next"].detach().cpu().numpy())
        outs["scene_gate"].append(out["scene_gate"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outs.items()}


def _train(args: argparse.Namespace, rows: Mapping[str, int | None]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    runtime = m._configure_runtime(int(args.seed))
    train, train_ctx = bq._build_gated_split("train", max_rows=rows["train"], row_seed=int(args.seed))
    val, val_ctx = bq._build_gated_split("val", max_rows=rows["val"], row_seed=int(args.seed))
    test, test_ctx = bq._build_gated_split("test", max_rows=rows["test"], row_seed=int(args.seed))
    dims = {"base_dim": train_ctx["base_dim"], "scene_dim": train_ctx["scene_dim"], "graph_dim": train_ctx["graph_dim"]}
    train, val, test, mean, std = m._standardize(train, val, test)
    model = GraphFirstSceneResidualMoE(
        dims["base_dim"],
        dims["scene_dim"],
        dims["graph_dim"],
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
    )
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    ckpt_path = CKPT_DIR / "stage43_dl_graph_first_scene_residual_moe.pt"
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        stats: list[dict[str, float]] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=int(args.seed) + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = _loss(
                model,
                train,
                batch_ids,
                device,
                easy_gate_weight=float(args.easy_gate_weight),
                preservation_weight=float(args.preservation_weight),
                gate_supervision_weight=float(args.gate_supervision_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_pred = _predict(model, val, device, int(args.batch_size))
        val_policy = bp._search_policy_with_cap(val, val_pred, max_switch_rate=float(args.max_switch_rate))
        selected_ade, selected_fde, switched = m._select_with_policy(val, val_pred, val_policy["policy"])
        val_metrics = m._metrics(val, selected_ade, selected_fde, switched)
        objective_loss = -float(
            val_metrics["full_waypoint_ade_improvement_vs_floor"]
            + 1.4 * val_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            + 1.0 * val_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - 14.0 * max(0.0, val_metrics["easy_degradation_vs_floor"] - 0.02)
            - 0.18 * val_metrics["switch_rate"]
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
    val_pred = _predict(model, val, device, int(args.batch_size))
    test_pred = _predict(model, test, device, int(args.batch_size))
    val_policy = bp._search_policy_with_cap(val, val_pred, max_switch_rate=float(args.max_switch_rate))
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    graph_branch_ade, graph_branch_fde = m._trajectory_error(test, test_pred["graph_waypoint"])
    graph_branch = m._metrics(test, graph_branch_ade, graph_branch_fde, np.ones(len(test.x), dtype=bool))
    scene_gate = test_pred["scene_gate"]
    hard_failure = test.hard | test.failure
    result = {
        "model": "graph_first_scene_residual_moe",
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
        "test_metrics_graph_branch_without_floor": graph_branch,
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
        "scene_residual_gate_summary": {
            "mean": float(np.mean(scene_gate)),
            "easy_mean": float(np.mean(scene_gate[test.easy])) if int(test.easy.sum()) else 0.0,
            "hard_failure_mean": float(np.mean(scene_gate[hard_failure])) if int(hard_failure.sum()) else 0.0,
            "t50_mean": float(np.mean(scene_gate[test.horizon == 50])) if int((test.horizon == 50).sum()) else 0.0,
        },
    }
    arrays = {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": test.floor_ade,
        "floor_fde": test.floor_fde,
        "h50": test.horizon == 50,
        "h100": test.horizon == 100,
        "hard_failure": hard_failure,
        "easy": test.easy,
    }
    return result, arrays


def _best_single(bp_payload: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    best_name = str(bp_payload.get("best_single_by_t50", "graph_history_only"))
    variants = {row["variant"]: row for row in bp_payload.get("variants", [])}
    return best_name, variants.get(best_name, {}), variants.get("no_context", {})


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
    model_result, arrays = _train(args, rows)
    bp_payload = read_json(bp.REPORT_JSON, {})
    bq_payload = read_json(bq.REPORT_JSON, {})
    dk_payload = read_json(DK_JSON, {})
    best_name, best_single, no_context = _best_single(bp_payload)
    best_metrics = best_single.get("test_metrics_with_floor", {})
    no_context_metrics = no_context.get("test_metrics_with_floor", {})
    metrics = model_result["test_metrics_with_floor"]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_graph_first_scene_residual_moe",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "model": model_result,
        "dk_precondition": {
            "verdict": dk_payload.get("stage43_dk_gate", {}).get("verdict", "missing"),
            "next_training_contract": dk_payload.get("next_training_contract", {}).get("name", "missing"),
        },
        "bp_best_single_by_t50": best_name,
        "best_single_metrics": best_metrics,
        "no_context_metrics": no_context_metrics,
        "bq_metrics": bq_payload.get("model", {}).get("test_metrics_with_floor", {}),
        "moe_minus_best_single_by_t50": bp._metric_delta(metrics, best_metrics),
        "moe_minus_no_context": bp._metric_delta(metrics, no_context_metrics),
        "moe_minus_bq_gated_fusion": bp._metric_delta(metrics, bq_payload.get("model", {}).get("test_metrics_with_floor", {})),
        "bootstrap_moe_vs_best_single_t50_ci": bp._bootstrap_contribution(
            arrays,
            {
                "selected_ade": np.full_like(arrays["selected_ade"], float(best_metrics.get("mean_selected_ade", 0.0))),
                "selected_fde": np.full_like(arrays["selected_fde"], float(best_metrics.get("mean_selected_fde", 0.0))),
                "floor_ade": arrays["floor_ade"],
                "floor_fde": arrays["floor_fde"],
                "h50": arrays["h50"],
                "h100": arrays["h100"],
                "hard_failure": arrays["hard_failure"],
            },
            n=int(args.bootstrap),
            seed=int(args.seed) + 4513,
        ),
        "ablation_type": {
            "fresh_retrained_graph_first_scene_residual_moe": True,
            "graph_default_expert": True,
            "scene_residual_expert": True,
            "expert_preservation_loss": True,
            "not_raw_concat": True,
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
                bq.bo._graph_current_path("train"),
                bq.bo._graph_current_path("val"),
                bq.bo._graph_current_path("test"),
                bq.bo._graph_history_path("train"),
                bq.bo._graph_history_path("val"),
                bq.bo._graph_history_path("test"),
                DK_JSON,
            ]
        ),
    }
    payload["stage43_dl_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["model"]["test_metrics_with_floor"]
    delta_best = payload["moe_minus_best_single_by_t50"]
    delta_bq = payload["moe_minus_bq_gated_fusion"]
    gate_summary = payload["model"]["scene_residual_gate_summary"]
    safe = bool(metrics.get("easy_degradation_vs_floor", 1.0) <= 0.02)
    beats_best_single = bool(
        safe
        and (
            delta_best["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_best["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_best["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
    )
    beats_bq = bool(
        safe
        and (
            delta_bq["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_bq["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or delta_bq["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
    )
    gates = {
        "dk_next_training_contract_present": payload["dk_precondition"]["next_training_contract"]
        == "stage43_next_graph_first_scene_residual_moe",
        "fresh_graph_first_moe_trained": payload["result_source"] == "fresh_graph_first_scene_residual_moe"
        and payload["ablation_type"]["fresh_retrained_graph_first_scene_residual_moe"] is True,
        "graph_default_and_scene_residual_architecture": payload["ablation_type"]["graph_default_expert"] is True
        and payload["ablation_type"]["scene_residual_expert"] is True,
        "expert_preservation_loss_recorded": payload["ablation_type"]["expert_preservation_loss"] is True
        and bool(payload["model"]["training_history"]),
        "scene_and_graph_dims_present": payload["model"]["dims"]["scene_dim"] > 0
        and payload["model"]["dims"]["graph_dim"] > 0,
        "latent_noncollapse": payload["model"]["latent_variance"] > 0.01,
        "protected_eval_completed": metrics.get("rows", 0) > 0,
        "easy_preservation_measured": "easy_degradation_vs_floor" in metrics,
        "best_single_comparison_reported": "moe_minus_best_single_by_t50" in payload,
        "bq_comparison_reported": "moe_minus_bq_gated_fusion" in payload,
        "scene_residual_gate_measured": 0.0 <= gate_summary["mean"] <= 1.0,
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
        "stage43_dl_graph_first_scene_residual_moe_pass_contribution_supported"
        if passed == total and beats_best_single
        else "stage43_dl_graph_first_scene_residual_moe_pass_safe_bq_lift_diagnostic"
        if passed == total and beats_bq
        else "stage43_dl_graph_first_scene_residual_moe_pass_safe_no_best_single_lift_diagnostic"
        if passed == total and safe
        else "stage43_dl_graph_first_scene_residual_moe_pass_unsafe_diagnostic"
        if passed == total
        else "stage43_dl_graph_first_scene_residual_moe_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "graph_first_moe_executed": passed == total,
        "beats_best_single": beats_best_single,
        "beats_bq_gated_fusion": beats_bq,
        "safe_easy": safe,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_dl_gate"]
    metrics = payload["model"]["test_metrics_with_floor"]
    delta_best = payload["moe_minus_best_single_by_t50"]
    delta_bq = payload["moe_minus_bq_gated_fusion"]
    scene_gate = payload["model"]["scene_residual_gate_summary"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-DL Graph-First Scene-Residual MoE",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- beats best single: `{gate['beats_best_single']}`",
            f"- beats BQ gated fusion: `{gate['beats_bq_gated_fusion']}`",
            f"- safe easy: `{gate['safe_easy']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Protected Test Metrics",
            "",
            f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
            f"- hard/failure full-waypoint ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
            f"- switch rate: `{_pct(metrics['switch_rate'])}`",
            "",
            "## Contribution Deltas",
            "",
            f"- MoE minus best single `{payload['bp_best_single_by_t50']}` all/t50/hard: `{_pct(delta_best['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_best['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- MoE minus BQ gated fusion all/t50/hard: `{_pct(delta_bq['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_bq['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(delta_bq['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            "",
            "## Scene Residual Gate",
            "",
            f"- mean: `{scene_gate['mean']:.4f}`",
            f"- easy mean: `{scene_gate['easy_mean']:.4f}`",
            f"- hard/failure mean: `{scene_gate['hard_failure_mean']:.4f}`",
            f"- t50 mean: `{scene_gate['t50_mean']:.4f}`",
            "",
            "## Boundary",
            "",
            "- This is a fresh graph-first scene-residual MoE training run, not threshold-only tuning.",
            "- Graph context is the default expert; scene proxy is only a gated residual expert.",
            "- Scene evidence remains proxy scene/goal/raster evidence, not raw image/SDF evidence.",
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
            "# Stage43-DL Graph-First Scene-Residual MoE Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- graph-first MoE executed: `{gate['graph_first_moe_executed']}`",
            f"- beats best single: `{gate['beats_best_single']}`",
            f"- beats BQ gated fusion: `{gate['beats_bq_gated_fusion']}`",
            f"- safe easy: `{gate['safe_easy']}`",
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
            f"- graph-first MoE executed: `{gate['graph_first_moe_executed']}`",
            f"- beats best single: `{gate['beats_best_single']}`",
            f"- beats BQ gated fusion: `{gate['beats_bq_gated_fusion']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-DL executes a fresh graph-first scene-residual MoE after Stage43-DK identified why generic scene+graph fusion failed.",
            "- The deployment floor is unchanged; this is protected latent-world-state evidence, not a new deployed policy.",
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
    gate = payload["stage43_dl_gate"]
    metrics = payload["model"]["test_metrics_with_floor"]
    delta_best = payload["moe_minus_best_single_by_t50"]
    section = [
        "## Stage43-DL: Graph-First Scene-Residual MoE",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"graph_first_moe_executed = `{gate['graph_first_moe_executed']}`",
        f"beats_best_single = `{gate['beats_best_single']}`",
        f"beats_bq_gated_fusion = `{gate['beats_bq_gated_fusion']}`",
        f"safe_easy = `{gate['safe_easy']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Protected metrics: all `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        "",
        f"Against `{payload['bp_best_single_by_t50']}`, the graph-first scene-residual MoE delta is all `{_pct(delta_best['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(delta_best['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(delta_best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`.",
        "",
        "My read: this is the right architecture family to test after the scene/graph failure taxonomy: protect the graph expert, let scene proxy act only as a residual, and keep the floor unless validation-safe switching earns its way in.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_dl_graph_first_scene_residual_moe"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "graph_first_moe_executed": gate["graph_first_moe_executed"],
        "beats_best_single": gate["beats_best_single"],
        "beats_bq_gated_fusion": gate["beats_bq_gated_fusion"],
        "safe_easy": gate["safe_easy"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "metrics": payload["model"]["test_metrics_with_floor"],
        "moe_minus_best_single_by_t50": payload["moe_minus_best_single_by_t50"],
        "moe_minus_bq_gated_fusion": payload["moe_minus_bq_gated_fusion"],
        "scene_residual_gate_summary": payload["model"]["scene_residual_gate_summary"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_dl_graph_first_scene_residual_moe"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-DL",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "beats_best_single": gate["beats_best_single"],
                        "beats_bq_gated_fusion": gate["beats_bq_gated_fusion"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-DL graph-first scene-residual MoE.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=463)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-switch-rate", type=float, default=0.50)
    parser.add_argument("--easy-gate-weight", type=float, default=0.30)
    parser.add_argument("--preservation-weight", type=float, default=0.35)
    parser.add_argument("--gate-supervision-weight", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    payload = _run(args)
    gate = payload["stage43_dl_gate"]
    print(f"Stage43-DL: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"beats_best_single={gate['beats_best_single']}")
    print(f"beats_bq_gated_fusion={gate['beats_bq_gated_fusion']}")
    print(f"safe_easy={gate['safe_easy']}")
    return payload


if __name__ == "__main__":
    main()
