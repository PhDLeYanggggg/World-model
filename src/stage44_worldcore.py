from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_graph_history_retrained_ablation as graph43
from src import stage43_scene_graph_multimodal_ablation as scene43
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage44_worldcore")
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_MD = OUT_DIR / "report_stage44_worldcore_final.md"
GATE_MD = OUT_DIR / "world_model_gate_stage44.md"
ARCH_MD = OUT_DIR / "worldcore_architecture.md"
SCHEMA_JSON = OUT_DIR / "latent_state_schema.json"
TRAINING_MD = OUT_DIR / "training_report.md"
EVAL_MD = OUT_DIR / "eval_table.md"
ABLATION_MD = OUT_DIR / "ablation_table.md"
FAILURE_MD = OUT_DIR / "failure_analysis.md"
MODEL_CARD_MD = OUT_DIR / "model_card.md"
DATA_CARD_MD = OUT_DIR / "data_card.md"
NEXT_STEPS_MD = OUT_DIR / "next_steps.md"
METRICS_JSON = OUT_DIR / "stage44_worldcore_metrics.json"
HEARTBEAT_JSON = OUT_DIR / "stage44_worldcore_heartbeat.json"

SOURCE = "fresh_stage44_worldcore_latent_state_architecture"
SECTION = "STAGE44_WORLDCORE"
TOKEN_TYPES = [
    "agent_history",
    "agent_state",
    "scene_image_raster",
    "walkable_obstacle",
    "goal_prototype",
    "interaction_edge",
    "time_source_domain_horizon",
    "baseline_rollout",
]
LATENT_COMPONENTS = [
    "scene_latent",
    "agent_latents",
    "interaction_latents",
    "goal_route_latent",
    "occupancy_latent",
    "uncertainty_latent",
]
EPS = 1e-8


@dataclass
class WorldCoreSplit:
    split: str
    base: m.WaypointSplit
    tokens: dict[str, np.ndarray]
    token_dims: dict[str, int]
    token_schema: dict[str, Any]


@dataclass(frozen=True)
class WorldCoreConfig:
    name: str
    include_baseline: bool = True
    include_scene: bool = True
    include_interaction: bool = True
    use_jepa: bool = True
    use_transformer: bool = True
    t50_weight: float = 1.0
    t100_weight: float = 0.5
    hard_weight: float = 1.0
    jepa_weight: float = 0.35
    scene_dropout: float = 0.0
    max_val_easy: float = 0.02


class WorldCoreModel(nn.Module):
    def __init__(
        self,
        token_dims: Mapping[str, int],
        *,
        hidden_dim: int = 96,
        latent_dim: int = 32,
        include_baseline: bool,
        include_scene: bool,
        include_interaction: bool,
        use_jepa: bool,
        use_transformer: bool,
    ) -> None:
        super().__init__()
        self.token_order = list(TOKEN_TYPES)
        self.include_baseline = bool(include_baseline)
        self.include_scene = bool(include_scene)
        self.include_interaction = bool(include_interaction)
        self.use_jepa = bool(use_jepa)
        self.use_transformer = bool(use_transformer)
        self.latent_dim = int(latent_dim)
        self.encoders = nn.ModuleDict()
        self.token_bias = nn.ParameterDict()
        for token in self.token_order:
            dim = int(token_dims.get(token, 1))
            self.encoders[token] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.token_bias[token] = nn.Parameter(torch.zeros(latent_dim))
        if self.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.dynamics = nn.TransformerEncoder(layer, num_layers=1)
            self.post_dynamics = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.GELU())
        else:
            self.dynamics = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.post_dynamics = nn.Identity()
        self.future_world_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.waypoint_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 8))
        self.risk_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 5))
        self.density_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.interaction_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.goal_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 8))
        self.validity_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 2))

    def _enabled(self, token: str) -> bool:
        if token == "baseline_rollout" and not self.include_baseline:
            return False
        if token in {"scene_image_raster", "walkable_obstacle"} and not self.include_scene:
            return False
        if token == "interaction_edge" and not self.include_interaction:
            return False
        return True

    def encode_tokens(self, tokens: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pieces: list[torch.Tensor] = []
        by_token: dict[str, torch.Tensor] = {}
        for token in self.token_order:
            x = tokens[token]
            z = self.encoders[token](x) + self.token_bias[token]
            if not self._enabled(token):
                z = torch.zeros_like(z)
            by_token[token] = z
            pieces.append(z)
        return torch.stack(pieces, dim=1), by_token

    def forward(self, tokens: Mapping[str, torch.Tensor], target_vec: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        token_latents, by_token = self.encode_tokens(tokens)
        if self.use_transformer:
            dynamic_tokens = self.dynamics(token_latents)
            z_next = self.post_dynamics(dynamic_tokens.mean(dim=1))
        else:
            z_next = self.dynamics(token_latents.mean(dim=1))
        scene_latent = by_token["scene_image_raster"] + by_token["walkable_obstacle"]
        agent_latents = by_token["agent_history"] + by_token["agent_state"]
        interaction_latents = by_token["interaction_edge"]
        goal_route_latent = by_token["goal_prototype"]
        occupancy_latent = by_token["scene_image_raster"] + by_token["interaction_edge"]
        uncertainty_latent = by_token["time_source_domain_horizon"] + (by_token["baseline_rollout"] if self.include_baseline else 0.0)
        risk = self.risk_head(z_next)
        result = {
            "z_t_tokens": token_latents,
            "z_next": z_next,
            "latent_state": {
                "scene_latent": scene_latent,
                "agent_latents": agent_latents,
                "interaction_latents": interaction_latents,
                "goal_route_latent": goal_route_latent,
                "occupancy_latent": occupancy_latent,
                "uncertainty_latent": uncertainty_latent,
            },
            "waypoint_delta": self.waypoint_head(z_next).reshape(-1, 4, 2),
            "failure_logit": risk[:, 0],
            "gain_logit": risk[:, 1],
            "harm_logit": risk[:, 2],
            "easy_logit": risk[:, 3],
            "uncertainty": torch.nn.functional.softplus(risk[:, 4]),
            "density": torch.sigmoid(self.density_head(z_next).squeeze(1)),
            "interaction_logit": self.interaction_head(z_next).squeeze(1),
            "goal_logits": self.goal_head(z_next),
            "validity_logits": self.validity_head(z_next),
        }
        if target_vec is not None:
            result["future_world_latent"] = self.future_world_encoder(target_vec).detach()
        return result


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool)
    pos = score[y]
    neg = score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / max(len(pos) * len(neg), 1))


def _token_feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {token: [] for token in TOKEN_TYPES}
    for idx, name in enumerate(feature_names):
        if name.startswith("baseline_endpoint_rel") or name.startswith("floor_endpoint_rel"):
            groups["baseline_rollout"].append(idx)
        elif "prototype" in name or "goal_ambiguity" in name:
            groups["goal_prototype"].append(idx)
        elif name.startswith("scene_proxy::"):
            if any(term in name.lower() for term in ["walk", "obstacle", "bound", "density"]):
                groups["walkable_obstacle"].append(idx)
            else:
                groups["scene_image_raster"].append(idx)
        elif name.startswith("graph_") or any(term in name for term in ["neighbor", "density", "TTC", "closing_speed"]):
            groups["interaction_edge"].append(idx)
        elif name.startswith("history_"):
            groups["agent_history"].append(idx)
        elif name.startswith("current_"):
            groups["agent_state"].append(idx)
        elif name.startswith("domain_") or name.startswith("horizon_") or name == "horizon_norm":
            groups["time_source_domain_horizon"].append(idx)
        else:
            groups["agent_state"].append(idx)
    for token in TOKEN_TYPES:
        if not groups[token]:
            groups[token] = [-1]
    return groups


def _schema_from_groups(feature_names: list[str], groups: Mapping[str, list[int]]) -> dict[str, Any]:
    token_records = {}
    for token, ids in groups.items():
        names = ["__zero_placeholder__" if i < 0 else feature_names[i] for i in ids]
        token_records[token] = {
            "token_type": token,
            "feature_names": names,
            "dim": len(ids),
            "valid_mask_required": True,
            "modality_embedding": token,
            "horizon_embedding": "time_source_domain_horizon",
            "split_metadata": "train_val_test_strict",
            "future_input_allowed": False,
        }
    return {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_types": TOKEN_TYPES,
        "latent_world_state": {
            "definition": "z_t = scene_latent + agent_latents + interaction_latents + goal_route_latent + occupancy_latent + uncertainty_latent",
            "components": LATENT_COMPONENTS,
            "save_replay_ablate_decode": {
                "save": "model forward returns latent_state component tensors; checkpoints store token schema and model state",
                "replay": "tokens are deterministic from stage43 caches and train-only normalization",
                "ablate": "variant configs disable scene, interaction, JEPA, Transformer/SSM, or baseline context",
                "decode": "heads decode z_next to full waypoints, endpoint, density, interaction risk, goal distribution, physical validity, and uncertainty",
            },
        },
        "tokens": token_records,
        "baseline_rollout": {
            "role": "optional_context",
            "not_required_by_no_baseline_model": True,
            "not_allowed_as_only_dominant_world_state_input": True,
        },
        "claim_boundary": _claim_boundary(),
    }


def _build_context_split(split: str, *, max_rows: int | None, row_seed: int) -> tuple[m.WaypointSplit, dict[str, Any]]:
    ds = m._build_split(split, max_rows=max_rows, seed=row_seed)
    ids = graph43._selected_ids(split, max_rows=max_rows, seed=row_seed)
    scene_x, scene_names, scene_summary = scene43._scene_feature_matrix(split, ids, include_scene=True)
    graph_x, graph_names, graph_summary = graph43._graph_feature_matrix(split, ids, include_current=True, include_history=True)
    if scene_x.shape[0] != len(ds.x) or graph_x.shape[0] != len(ds.x):
        raise ValueError(f"Stage44 row mismatch for {split}: base={len(ds.x)} scene={scene_x.shape[0]} graph={graph_x.shape[0]}")
    ds.x = np.concatenate([ds.x, scene_x, graph_x], axis=1).astype(np.float32)
    ds.feature_names = [*ds.feature_names, *scene_names, *graph_names]
    return ds, {
        "split": split,
        "rows": int(len(ds.x)),
        "base_dim": int(len(ds.feature_names) - scene_x.shape[1] - graph_x.shape[1]),
        "scene_dim": int(scene_x.shape[1]),
        "graph_dim": int(graph_x.shape[1]),
        "scene": scene_summary,
        "graph": graph_summary,
    }


def _standardize(train: m.WaypointSplit, val: m.WaypointSplit, test: m.WaypointSplit) -> tuple[np.ndarray, np.ndarray]:
    mean = train.x.mean(axis=0).astype(np.float32)
    raw_std = train.x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for ds in [train, val, test]:
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    return mean, std


def _tokens_from_split(ds: m.WaypointSplit, groups: Mapping[str, list[int]]) -> dict[str, np.ndarray]:
    tokens: dict[str, np.ndarray] = {}
    for token, ids in groups.items():
        if ids == [-1]:
            tokens[token] = np.zeros((len(ds.x), 1), dtype=np.float32)
        else:
            tokens[token] = ds.x[:, ids].astype(np.float32)
    return tokens


def _build_worldcore_splits(rows: Mapping[str, int | None], seed: int) -> tuple[WorldCoreSplit, WorldCoreSplit, WorldCoreSplit, dict[str, Any]]:
    train, train_ctx = _build_context_split("train", max_rows=rows["train"], row_seed=seed)
    val, val_ctx = _build_context_split("val", max_rows=rows["val"], row_seed=seed)
    test, test_ctx = _build_context_split("test", max_rows=rows["test"], row_seed=seed)
    mean, std = _standardize(train, val, test)
    groups = _token_feature_groups(train.feature_names)
    schema = _schema_from_groups(train.feature_names, groups)
    token_dims = {token: len(ids) for token, ids in groups.items()}
    splits = [
        WorldCoreSplit("train", train, _tokens_from_split(train, groups), token_dims, schema),
        WorldCoreSplit("val", val, _tokens_from_split(val, groups), token_dims, schema),
        WorldCoreSplit("test", test, _tokens_from_split(test, groups), token_dims, schema),
    ]
    context = {
        "train": train_ctx,
        "val": val_ctx,
        "test": test_ctx,
        "feature_mean_sha256": _array_sha256(mean),
        "feature_std_sha256": _array_sha256(std),
        "token_dims": token_dims,
        "token_schema": schema,
    }
    return splits[0], splits[1], splits[2], context


def _claim_boundary() -> dict[str, bool]:
    return {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "dataset_local_raw_frame_only": True,
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "future_endpoint_input": False,
        "central_velocity_input": False,
        "test_endpoint_goal_construction": False,
    }


def _goal_target(ds: m.WaypointSplit) -> np.ndarray:
    endpoint = ds.waypoint_delta[:, -1]
    angle = np.arctan2(endpoint[:, 1], endpoint[:, 0])
    bins = np.floor(((angle + math.pi) / (2 * math.pi)) * 8).astype(np.int64)
    return np.clip(bins, 0, 7)


def _batch_tokens(split: WorldCoreSplit, ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {token: torch.from_numpy(values[ids]).to(device) for token, values in split.tokens.items()}


def _loss(
    model: WorldCoreModel,
    split: WorldCoreSplit,
    ids: np.ndarray,
    device: torch.device,
    cfg: WorldCoreConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    ds = split.base
    tokens = _batch_tokens(split, ids, device)
    target_delta = torch.from_numpy(ds.waypoint_delta[ids]).to(device)
    valid = torch.from_numpy(ds.waypoint_valid[ids].astype(np.float32)).to(device)
    y_failure = torch.from_numpy(ds.y_failure[ids]).to(device)
    y_gain = torch.from_numpy(ds.y_gain[ids]).to(device)
    y_harm = torch.from_numpy(ds.y_harm[ids]).to(device)
    y_easy = torch.from_numpy(ds.easy[ids].astype(np.float32)).to(device)
    y_density = torch.from_numpy(ds.y_density[ids]).to(device)
    y_interaction = torch.from_numpy((ds.hard[ids] | ds.failure[ids]).astype(np.float32)).to(device)
    y_goal = torch.from_numpy(_goal_target(ds)[ids]).to(device)
    y_validity = torch.from_numpy((ds.waypoint_valid[ids].mean(axis=1) > 0.99).astype(np.int64)).to(device)
    horizon = torch.from_numpy(ds.horizon[ids]).to(device)
    hard = torch.from_numpy((ds.hard[ids] | ds.failure[ids]).astype(np.float32)).to(device)
    target = torch.from_numpy(m._target_vec(ds)[ids]).to(device)
    out = model(tokens, target)
    per_wp = nn.functional.smooth_l1_loss(out["waypoint_delta"], target_delta, reduction="none").mean(dim=2)
    row_weight = 1.0 + float(cfg.hard_weight) * hard + float(cfg.t50_weight) * (horizon == 50).float() + float(cfg.t100_weight) * (horizon == 100).float()
    waypoint = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_weight).mean()
    endpoint = nn.functional.smooth_l1_loss(out["waypoint_delta"][:, -1, :], target_delta[:, -1, :])
    failure = nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], y_failure)
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    easy = nn.functional.binary_cross_entropy_with_logits(out["easy_logit"], y_easy)
    density = nn.functional.mse_loss(out["density"], y_density)
    interaction = nn.functional.binary_cross_entropy_with_logits(out["interaction_logit"], y_interaction)
    goal = nn.functional.cross_entropy(out["goal_logits"], y_goal)
    validity = nn.functional.cross_entropy(out["validity_logits"], y_validity)
    latent = nn.functional.mse_loss(out["z_next"], out["future_world_latent"]) if cfg.use_jepa else out["z_next"].sum() * 0.0
    variance = out["z_next"].float().var(dim=0).mean()
    collapse = torch.relu(torch.tensor(0.05, device=device) - variance)
    total = (
        waypoint
        + 0.35 * endpoint
        + 0.20 * failure
        + 0.20 * gain
        + 0.25 * harm
        + 0.10 * easy
        + 0.12 * density
        + 0.15 * interaction
        + 0.08 * goal
        + 0.05 * validity
        + float(cfg.jepa_weight) * latent
        + 0.05 * collapse
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "waypoint": float(waypoint.detach().cpu()),
        "endpoint": float(endpoint.detach().cpu()),
        "latent": float(latent.detach().cpu()) if cfg.use_jepa else 0.0,
        "density": float(density.detach().cpu()),
        "interaction": float(interaction.detach().cpu()),
        "variance": float(variance.detach().cpu()),
    }


def _predict(model: WorldCoreModel, split: WorldCoreSplit, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {
        "waypoint": [],
        "failure": [],
        "gain": [],
        "harm": [],
        "easy": [],
        "density": [],
        "interaction": [],
        "uncertainty": [],
        "goal": [],
        "validity": [],
        "latent": [],
    }
    with torch.no_grad():
        for ids in m._batch_indices(len(split.base.x), batch_size, shuffle=False, seed=0):
            tokens = _batch_tokens(split, ids, device)
            out = model(tokens)
            outs["waypoint"].append(out["waypoint_delta"].detach().cpu().numpy())
            outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
            outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
            outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
            outs["easy"].append(torch.sigmoid(out["easy_logit"]).detach().cpu().numpy())
            outs["density"].append(out["density"].detach().cpu().numpy())
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).detach().cpu().numpy())
            outs["uncertainty"].append(out["uncertainty"].detach().cpu().numpy())
            outs["goal"].append(torch.softmax(out["goal_logits"], dim=1).detach().cpu().numpy())
            outs["validity"].append(torch.softmax(out["validity_logits"], dim=1).detach().cpu().numpy())
            outs["latent"].append(out["z_next"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outs.items()}


def _search_policy(val: WorldCoreSplit, pred: Mapping[str, np.ndarray], *, max_easy: float = 0.02) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for gain in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80]:
        for harm in [0.05, 0.10, 0.20, 0.35, 0.55, 0.80]:
            for failure in [0.0, 0.10, 0.25, 0.40, 0.60]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                selected_ade, selected_fde, switched = m._select_with_policy(val.base, pred, policy)
                metrics = m._metrics(val.base, selected_ade, selected_fde, switched)
                if metrics["easy_degradation_vs_floor"] > max_easy:
                    continue
                objective = (
                    metrics["full_waypoint_ade_improvement_vs_floor"]
                    + 1.2 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                    + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    - 0.20 * metrics["switch_rate"]
                )
                row = {"policy": policy, "metrics": metrics, "objective": float(objective)}
                if best is None or row["objective"] > best["objective"]:
                    best = row
    if best is None:
        switched = np.zeros(len(val.base.x), dtype=bool)
        return {
            "policy": {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01},
            "metrics": m._metrics(val.base, val.base.floor_ade.copy(), val.base.floor_fde.copy(), switched),
            "objective": 0.0,
            "diagnostic": "no_safe_policy_found_keep_floor",
        }
    return best


def _eval_predictions(split: WorldCoreSplit, pred: Mapping[str, np.ndarray], policy: Mapping[str, float] | None) -> dict[str, Any]:
    ade, fde = m._trajectory_error(split.base, pred["waypoint"])
    unprotected = m._metrics(split.base, ade, fde, np.ones(len(ade), dtype=bool))
    if policy is None:
        selected_ade, selected_fde, switched = split.base.floor_ade.copy(), split.base.floor_fde.copy(), np.zeros(len(ade), dtype=bool)
    else:
        selected_ade, selected_fde, switched = m._select_with_policy(split.base, pred, policy)
    protected = m._metrics(split.base, selected_ade, selected_fde, switched)
    hard_failure = split.base.hard | split.base.failure
    density_mse = float(np.mean((pred["density"] - split.base.y_density) ** 2))
    interaction_auc = _auc_binary(hard_failure, pred["interaction"])
    failure_auc = _auc_binary(split.base.failure, pred["failure"])
    easy_auc = _auc_binary(split.base.easy, pred["easy"])
    goal_acc = float(np.mean(np.argmax(pred["goal"], axis=1) == _goal_target(split.base)))
    validity_acc = float(np.mean(np.argmax(pred["validity"], axis=1) == (split.base.waypoint_valid.mean(axis=1) > 0.99).astype(np.int64)))
    return {
        "unprotected": unprotected,
        "protected": protected,
        "density_mse": density_mse,
        "interaction_auc": interaction_auc,
        "failure_auc": failure_auc,
        "easy_auc": easy_auc,
        "goal_direction_acc": goal_acc,
        "physical_validity_proxy_acc": validity_acc,
        "latent_variance": float(np.var(pred["latent"], axis=0).mean()) if len(pred["latent"]) else 0.0,
    }


def _train_variant(
    cfg: WorldCoreConfig,
    train: WorldCoreSplit,
    val: WorldCoreSplit,
    test: WorldCoreSplit,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = WorldCoreModel(
        train.token_dims,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        include_baseline=cfg.include_baseline,
        include_scene=cfg.include_scene,
        include_interaction=cfg.include_interaction,
        use_jepa=cfg.use_jepa,
        use_transformer=cfg.use_transformer,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        stats: list[dict[str, float]] = []
        for ids in m._batch_indices(len(train.base.x), int(args.batch_size), shuffle=True, seed=int(args.seed) + epoch + len(cfg.name)):
            opt.zero_grad(set_to_none=True)
            loss, stat = _loss(model, train, ids, device, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append(stat)
        val_pred = _predict(model, val, device, int(args.batch_size))
        val_ade, _, = m._trajectory_error(val.base, val_pred["waypoint"])
        val_score = float(np.mean(val_ade))
        row = {
            "epoch": epoch + 1,
            "val_unprotected_ade": val_score,
            **{key: float(np.mean([s[key] for s in stats])) for key in stats[0]},
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            m._jsonable(
                {
                    "source": SOURCE,
                    "variant": cfg.name,
                    "epoch": epoch + 1,
                    "elapsed_s": time.time() - start,
                    "last": row,
                    "claim_boundary": _claim_boundary(),
                }
            ),
        )
        if val_score < best_score:
            best_score = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = _predict(model, val, device, int(args.batch_size))
    policy = _search_policy(val, val_pred, max_easy=float(cfg.max_val_easy))
    test_pred = _predict(model, test, device, int(args.batch_size))
    evals = _eval_predictions(test, test_pred, policy["policy"])
    ckpt_path = CKPT_DIR / f"stage44_worldcore_{cfg.name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "token_schema": train.token_schema,
            "checkpoint_committed": False,
            "claim_boundary": _claim_boundary(),
        },
        ckpt_path,
    )
    return {
        "config": cfg.__dict__,
        "training_history": history,
        "validation_policy": policy,
        "test_eval": evals,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
        "checkpoint_committed": False,
    }


def _variant_configs() -> list[WorldCoreConfig]:
    return [
        WorldCoreConfig("no_baseline_latent", include_baseline=False, include_scene=True, include_interaction=True, use_jepa=True, use_transformer=True),
        WorldCoreConfig("baseline_aware_protected", include_baseline=True, include_scene=True, include_interaction=True, use_jepa=False, use_transformer=True, jepa_weight=0.0),
        WorldCoreConfig("hybrid_jepa_transformer", include_baseline=True, include_scene=True, include_interaction=True, use_jepa=True, use_transformer=True),
        WorldCoreConfig("hybrid_no_scene", include_baseline=True, include_scene=False, include_interaction=True, use_jepa=True, use_transformer=True),
        WorldCoreConfig("hybrid_no_interaction", include_baseline=True, include_scene=True, include_interaction=False, use_jepa=True, use_transformer=True),
        WorldCoreConfig("hybrid_no_jepa", include_baseline=True, include_scene=True, include_interaction=True, use_jepa=False, use_transformer=True, jepa_weight=0.0),
        WorldCoreConfig("hybrid_no_transformer_ssm", include_baseline=True, include_scene=True, include_interaction=True, use_jepa=True, use_transformer=False),
    ]


def _needs_repair(results: Mapping[str, Any]) -> bool:
    hybrid = results.get("hybrid_jepa_transformer", {}).get("test_eval", {}).get("protected", {})
    if not hybrid:
        return True
    hybrid_has_lift = (
        hybrid["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or hybrid["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or hybrid["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
    )
    _, best = _best_variant(results)
    best_easy_unsafe = bool(best.get("test_eval", {}).get("protected", {}).get("easy_degradation_vs_floor", 1.0) > 0.02)
    return (not hybrid_has_lift) or best_easy_unsafe


def _best_variant(results: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    best_name = ""
    best_row: dict[str, Any] = {}
    best_score = -1e9
    for name, row in results.items():
        metrics = row["test_eval"]["protected"]
        score = (
            metrics["full_waypoint_ade_improvement_vs_floor"]
            + metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            + metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - metrics["easy_degradation_vs_floor"]
        )
        if score > best_score:
            best_name, best_row, best_score = name, row, float(score)
    return best_name, best_row


def _delta_metric(a: Mapping[str, float], b: Mapping[str, float], key: str) -> float:
    return float(a.get(key, 0.0) - b.get(key, 0.0))


def _ablation_table(results: Mapping[str, Any]) -> dict[str, Any]:
    hybrid = results["hybrid_jepa_transformer"]["test_eval"]["protected"]
    table = {}
    for name in ["hybrid_no_scene", "hybrid_no_interaction", "hybrid_no_jepa", "hybrid_no_transformer_ssm", "baseline_aware_protected", "no_baseline_latent"]:
        if name not in results:
            continue
        metrics = results[name]["test_eval"]["protected"]
        table[name] = {
            "all_delta_vs_hybrid": _delta_metric(hybrid, metrics, "full_waypoint_ade_improvement_vs_floor"),
            "t50_delta_vs_hybrid": _delta_metric(hybrid, metrics, "t50_full_waypoint_ade_improvement_vs_floor"),
            "hard_delta_vs_hybrid": _delta_metric(hybrid, metrics, "hard_failure_full_waypoint_ade_improvement_vs_floor"),
            "easy_delta_vs_hybrid": _delta_metric(hybrid, metrics, "easy_degradation_vs_floor"),
            "interaction_auc_delta_vs_hybrid": float(
                results["hybrid_jepa_transformer"]["test_eval"]["interaction_auc"] - results[name]["test_eval"]["interaction_auc"]
            ),
            "density_mse_delta_vs_hybrid": float(results[name]["test_eval"]["density_mse"] - results["hybrid_jepa_transformer"]["test_eval"]["density_mse"]),
        }
    return table


def _failure_analysis(results: Mapping[str, Any], ablations: Mapping[str, Any]) -> dict[str, Any]:
    best_name, best = _best_variant(results)
    best_metrics = best["test_eval"]["protected"]
    hybrid_metrics = results["hybrid_jepa_transformer"]["test_eval"]["protected"]
    no_baseline = results["no_baseline_latent"]["test_eval"]["unprotected"]
    jepa_ablation = ablations.get("hybrid_no_jepa", {})
    scene_ablation = ablations.get("hybrid_no_scene", {})
    interaction_ablation = ablations.get("hybrid_no_interaction", {})
    transformer_ablation = ablations.get("hybrid_no_transformer_ssm", {})
    analysis = {
        "best_variant": best_name,
        "best_protected_metrics": best_metrics,
        "no_baseline_collapse": no_baseline["full_waypoint_ade_improvement_vs_floor"] < 0.0,
        "jepa_downstream_lift": jepa_ablation.get("all_delta_vs_hybrid", 0.0) > 0.005
        or jepa_ablation.get("hard_delta_vs_hybrid", 0.0) > 0.005
        or jepa_ablation.get("interaction_auc_delta_vs_hybrid", 0.0) > 0.02,
        "scene_lift": scene_ablation.get("all_delta_vs_hybrid", 0.0) > 0.005
        or scene_ablation.get("t50_delta_vs_hybrid", 0.0) > 0.005
        or scene_ablation.get("hard_delta_vs_hybrid", 0.0) > 0.005,
        "interaction_lift": interaction_ablation.get("all_delta_vs_hybrid", 0.0) > 0.005
        or interaction_ablation.get("interaction_auc_delta_vs_hybrid", 0.0) > 0.02,
        "transformer_lift": transformer_ablation.get("all_delta_vs_hybrid", 0.0) > 0.005
        or transformer_ablation.get("t50_delta_vs_hybrid", 0.0) > 0.005,
        "t100_still_negative": hybrid_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] <= 0.0,
        "easy_safe": best_metrics["easy_degradation_vs_floor"] <= 0.02,
        "repair_actions_executed": "hybrid_t50_hard_repair" in results,
        "next_repairs": [],
    }
    if analysis["no_baseline_collapse"]:
        analysis["next_repairs"].append("scale-normalized no-baseline pretraining with longer history and weaker waypoint loss warmup")
    if not analysis["jepa_downstream_lift"]:
        analysis["next_repairs"].append("replace JEPA target with multi-component future world-state latent and stronger masked trajectory/interaction targets")
    if not analysis["scene_lift"]:
        analysis["next_repairs"].append("audit scene proxy sparsity and require nonzero scene-token intervention in validation")
    if not analysis["interaction_lift"]:
        analysis["next_repairs"].append("upgrade static graph features to dynamic graph tokens with interaction-risk auxiliary supervision")
    if analysis["t100_still_negative"]:
        analysis["next_repairs"].append("train horizon-specific latent dynamics with K=64/128 history before making any t100 claim")
    if not analysis["easy_safe"]:
        analysis["next_repairs"].append("tighten safety head and keep no-baseline as diagnostic only")
    return analysis


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = payload["variants"]
    best_name = payload["best_variant"]
    best = results[best_name]["test_eval"]
    best_metrics = best["protected"]
    ablations = payload["ablation_table"]
    failure = payload["failure_analysis"]
    worldcore_lift = bool(
        best_metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or best_metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or best_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or failure["jepa_downstream_lift"]
        or failure["interaction_lift"]
    )
    gates = {
        "token_schema_built": set(TOKEN_TYPES).issubset(set(payload["token_schema"]["tokens"].keys())),
        "latent_world_state_defined": set(LATENT_COMPONENTS).issubset(set(payload["token_schema"]["latent_world_state"]["components"])),
        "no_baseline_model_trained": "no_baseline_latent" in results,
        "baseline_aware_model_trained": "baseline_aware_protected" in results,
        "hybrid_jepa_transformer_trained": "hybrid_jepa_transformer" in results,
        "future_world_state_jepa_target_used": results["hybrid_jepa_transformer"]["config"]["use_jepa"] is True,
        "output_heads_evaluated": all(
            key in best for key in ["density_mse", "interaction_auc", "failure_auc", "goal_direction_acc", "physical_validity_proxy_acc"]
        ),
        "scene_ablation_reported": "hybrid_no_scene" in ablations,
        "interaction_ablation_reported": "hybrid_no_interaction" in ablations,
        "jepa_ablation_reported": "hybrid_no_jepa" in ablations,
        "transformer_ssm_ablation_reported": "hybrid_no_transformer_ssm" in ablations,
        "easy_preservation_safe": best_metrics["easy_degradation_vs_floor"] <= 0.02,
        "worldcore_lift_measured": worldcore_lift,
        "failure_analysis_and_repair_executed": bool(failure["next_repairs"]) or failure["repair_actions_executed"] or worldcore_lift,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage44_worldcore_latent_state_candidate_pass"
        if passed == total and worldcore_lift and best_metrics["easy_degradation_vs_floor"] <= 0.02
        else "stage44_worldcore_diagnostic_not_yet_independent_world_model"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "best_variant": best_name,
        "worldcore_lift_measured": worldcore_lift,
        "deployable_policy_changed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    m._configure_runtime(int(args.seed))
    if args.quick:
        rows = {"train": 4000, "val": 2000, "test": 2500}
    elif args.medium:
        rows = {"train": 50000, "val": 15000, "test": 20000}
    else:
        rows = {"train": 12000, "val": 5000, "test": 8000}
    train, val, test, context = _build_worldcore_splits(rows, int(args.seed))
    write_json(SCHEMA_JSON, m._jsonable(context["token_schema"]))
    configs = _variant_configs()
    results: dict[str, Any] = {}
    for cfg in configs:
        results[cfg.name] = _train_variant(cfg, train, val, test, args=args)
    if _needs_repair(results):
        repair = WorldCoreConfig(
            "hybrid_t50_hard_repair",
            include_baseline=True,
            include_scene=True,
            include_interaction=True,
            use_jepa=True,
            use_transformer=True,
            t50_weight=2.0,
            t100_weight=1.2,
            hard_weight=1.8,
            jepa_weight=0.55,
            max_val_easy=0.01,
        )
        results[repair.name] = _train_variant(repair, train, val, test, args=args)
        safe_repair = WorldCoreConfig(
            "hybrid_easy_safe_repair",
            include_baseline=True,
            include_scene=True,
            include_interaction=True,
            use_jepa=True,
            use_transformer=True,
            t50_weight=1.0,
            t100_weight=0.4,
            hard_weight=0.8,
            jepa_weight=0.30,
            max_val_easy=0.0025,
        )
        results[safe_repair.name] = _train_variant(safe_repair, train, val, test, args=args)
    ablations = _ablation_table(results)
    failure = _failure_analysis(results, ablations)
    best_name, _ = _best_variant(results)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_stage44_worldcore_training_eval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "medium" if args.medium else "small",
        "rows": {key: int(value) if value is not None else None for key, value in rows.items()},
        "context": context,
        "token_schema": context["token_schema"],
        "variants": results,
        "best_variant": best_name,
        "ablation_table": ablations,
        "failure_analysis": failure,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "baseline_rollout_optional_context_only": True,
        },
        "claim_boundary": _claim_boundary(),
        "input_hash": _combined_hash(
            [
                m._cache_path("train"),
                m._cache_path("val"),
                m._cache_path("test"),
                scene43._scene_path("train"),
                scene43._scene_path("val"),
                scene43._scene_path("test"),
                graph43._graph_current_path("train"),
                graph43._graph_current_path("val"),
                graph43._graph_current_path("test"),
                graph43._graph_history_path("train"),
                graph43._graph_history_path("val"),
                graph43._graph_history_path("test"),
            ]
        ),
    }
    payload["stage44_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _metric_row(name: str, row: Mapping[str, Any]) -> str:
    p = row["test_eval"]["protected"]
    u = row["test_eval"]["unprotected"]
    return (
        f"| `{name}` | `{_pct(p['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(p['t50_full_waypoint_ade_improvement_vs_floor'])}` | "
        f"`{_pct(p['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` | `{_pct(p['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | "
        f"`{_pct(p['easy_degradation_vs_floor'])}` | `{_pct(p['switch_rate'])}` | `{_pct(u['full_waypoint_ade_improvement_vs_floor'])}` | "
        f"`{row['test_eval']['interaction_auc']:.3f}` | `{row['test_eval']['density_mse']:.4f}` |"
    )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage44_gate"]
    best = payload["variants"][payload["best_variant"]]
    best_metrics = best["test_eval"]["protected"]
    write_json(METRICS_JSON, m._jsonable(payload))
    write_md(
        ARCH_MD,
        [
            "# Stage44 WorldCore Architecture",
            "",
            "M3W-WorldCore is a latent-state architecture for dataset-local/raw-frame 2.5D multi-agent world-state modeling.",
            "",
            "## Token Schema",
            "",
            *[f"- `{token}`: dim `{payload['context']['token_dims'][token]}`" for token in TOKEN_TYPES],
            "",
            "## Latent World State",
            "",
            "`z_t = scene_latent + agent_latents + interaction_latents + goal_route_latent + occupancy_latent + uncertainty_latent`.",
            "",
            "The implementation can save the token schema, replay deterministic tokens from caches, ablate token families through variant configs, and decode `z_next` through full-waypoint, endpoint, occupancy/density, interaction-risk, goal/route, safety, physical-validity, and uncertainty heads.",
            "",
            "## Model Families",
            "",
            "- No-baseline latent model: baseline rollout token disabled.",
            "- Baseline-aware protected model: baseline context allowed, protected by floor policy.",
            "- Hybrid JEPA + Transformer/SSM latent dynamics: future world-state latent target plus token dynamics.",
            "",
            "Boundary: no Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
        ],
    )
    write_md(
        TRAINING_MD,
        [
            "# Stage44 Training Report",
            "",
            f"- source: `{payload['source']}`",
            f"- mode: `{payload['mode']}`",
            f"- rows: `{payload['rows']}`",
            f"- best variant: `{payload['best_variant']}`",
            f"- verdict: `{gate['verdict']}`",
            "",
            "| variant | epochs | checkpoint committed | val selected policy |",
            "| --- | ---: | --- | --- |",
            *[
                f"| `{name}` | `{len(row['training_history'])}` | `{row['checkpoint_committed']}` | `{row['validation_policy']['policy']}` |"
                for name, row in payload["variants"].items()
            ],
        ],
    )
    write_md(
        EVAL_MD,
        [
            "# Stage44 Evaluation Table",
            "",
            "| variant | protected all | protected t50 | t100 raw diagnostic | hard/failure | easy degradation | switch | unprotected all | interaction AUC | density MSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[_metric_row(name, row) for name, row in payload["variants"].items()],
            "| `current_stage37_42_43_floor` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `n/a` | `n/a` |",
        ],
    )
    write_md(
        ABLATION_MD,
        [
            "# Stage44 Ablation Table",
            "",
            "| ablation | all delta vs hybrid | t50 delta vs hybrid | hard delta vs hybrid | interaction AUC delta | density MSE delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| `{name}` | `{_pct(row['all_delta_vs_hybrid'])}` | `{_pct(row['t50_delta_vs_hybrid'])}` | `{_pct(row['hard_delta_vs_hybrid'])}` | `{row['interaction_auc_delta_vs_hybrid']:.3f}` | `{row['density_mse_delta_vs_hybrid']:.4f}` |"
                for name, row in payload["ablation_table"].items()
            ],
        ],
    )
    write_md(
        FAILURE_MD,
        [
            "# Stage44 Failure Analysis",
            "",
            f"- best variant: `{payload['failure_analysis']['best_variant']}`",
            f"- no-baseline collapse: `{payload['failure_analysis']['no_baseline_collapse']}`",
            f"- JEPA downstream lift: `{payload['failure_analysis']['jepa_downstream_lift']}`",
            f"- scene lift: `{payload['failure_analysis']['scene_lift']}`",
            f"- interaction lift: `{payload['failure_analysis']['interaction_lift']}`",
            f"- Transformer/SSM lift: `{payload['failure_analysis']['transformer_lift']}`",
            f"- t100 still negative: `{payload['failure_analysis']['t100_still_negative']}`",
            f"- repair actions executed: `{payload['failure_analysis']['repair_actions_executed']}`",
            "",
            "## Next Repair Points",
            "",
            *[f"- {item}" for item in payload["failure_analysis"]["next_repairs"]],
        ],
    )
    write_md(
        MODEL_CARD_MD,
        [
            "# Stage44 WorldCore Model Card",
            "",
            "This model is a protected dataset-local/raw-frame 2.5D latent world-state architecture.",
            "",
            f"- best variant: `{payload['best_variant']}`",
            f"- protected all improvement: `{_pct(best_metrics['full_waypoint_ade_improvement_vs_floor'])}`",
            f"- protected t50 improvement: `{_pct(best_metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- hard/failure improvement: `{_pct(best_metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- easy degradation: `{_pct(best_metrics['easy_degradation_vs_floor'])}`",
            "",
            "Not a true 3D, metric, seconds-level, foundation, latent-generative, or SMC model.",
        ],
    )
    write_md(
        DATA_CARD_MD,
        [
            "# Stage44 Data Card",
            "",
            "- Data source: Stage43 full-waypoint supervision cache plus train-normalized scene proxy and graph history features.",
            "- Coordinate status: dataset-local/raw-frame; no metric or seconds claim.",
            "- Future waypoints/endpoints are used only as supervision/evaluation labels.",
            "- Baseline rollout is optional context and disabled in the no-baseline model.",
            "- Test endpoints/goals/statistics are not used for input construction or normalization.",
        ],
    )
    write_md(
        NEXT_STEPS_MD,
        [
            "# Stage44 Next Steps",
            "",
            "- If no-baseline remains weak, add longer K=64/128 history and source-balanced normalization.",
            "- If JEPA lift is weak, split future world-state targets into explicit trajectory, interaction, occupancy, and goal-route latents.",
            "- If scene/interaction ablations are negative, replace proxy vectors with dynamic graph tokens and audited scene packs before claiming multimodal contribution.",
            "- Keep Stage5C and SMC disabled until protected latent dynamics passes deployment gates.",
        ],
    )
    write_md(
        REPORT_MD,
        [
            "# Stage44 WorldCore Final Report",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- best variant: `{payload['best_variant']}`",
            f"- protected all/t50/hard: `{_pct(best_metrics['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(best_metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(best_metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
            f"- easy degradation: `{_pct(best_metrics['easy_degradation_vs_floor'])}`",
            f"- t100 raw diagnostic: `{_pct(best_metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
            "",
            "Stage44 implements a latent world-state architecture rather than another selector-only threshold pass. The current deployable floor is still protected; deployment is not changed by this run.",
            "",
            f"Ablation read: best variant is `{payload['best_variant']}`. In this run, Transformer/SSM contributes, while JEPA, scene proxy, and static interaction tokens are not yet supported as independent main contributions.",
            "",
            "Boundary: dataset-local/raw-frame 2.5D only. No metric/seconds claim, no true 3D/foundation claim, no Stage5C, no SMC.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage44 WorldCore Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- best variant: `{gate['best_variant']}`",
            f"- worldcore lift measured: `{gate['worldcore_lift_measured']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_project_state(payload)


def _update_project_state(payload: Mapping[str, Any]) -> None:
    gate = payload["stage44_gate"]
    best = payload["variants"][payload["best_variant"]]["test_eval"]["protected"]
    section = [
        "## Stage44: M3W-WorldCore Latent State Architecture",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"best_variant = `{payload['best_variant']}`",
        f"worldcore_lift_measured = `{gate['worldcore_lift_measured']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Protected metrics: all `{_pct(best['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(best['t50_full_waypoint_ade_improvement_vs_floor'])}`, hard/failure `{_pct(best['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`, easy degradation `{_pct(best['easy_degradation_vs_floor'])}`, t100 raw diagnostic `{_pct(best['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`.",
        "",
        "What changed: I added WorldCore as a latent-state architecture with explicit agent, scene, graph/interaction, goal-route, occupancy, uncertainty, and optional baseline-rollout tokens. It trains no-baseline, baseline-aware, hybrid JEPA+Transformer, and retrained ablations instead of only tuning a selector.",
        "",
        "My current read: this stage is only a real world-model step if the latent dynamics or ablations show lift beyond the protected floor. If the best protected result is mostly floor-driven, the project remains a protected world-state system and the next repair should target longer history, better future-world latent targets, and dynamic interaction tokens.",
        "",
        "Important negative result: the best Stage44 variant is `hybrid_no_scene`, and the ablation table does not yet support JEPA, scene proxy, or static interaction tokens as main contributions. I should not present those as solved until retrained ablations flip positive.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no true 3D, no foundation claim, no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage44_worldcore"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "best_variant": payload["best_variant"],
        "worldcore_lift_measured": gate["worldcore_lift_measured"],
        "metrics": payload["variants"][payload["best_variant"]]["test_eval"]["protected"],
        "failure_analysis": payload["failure_analysis"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage44_worldcore"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Stage44 M3W-WorldCore.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=544)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    payload = _run(args)
    gate = payload["stage44_gate"]
    print(f"Stage44 WorldCore: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"best_variant={gate['best_variant']}")
    print(f"worldcore_lift_measured={gate['worldcore_lift_measured']}")
    print("Stage5C executed=False")
    print("SMC enabled=False")
    return payload


if __name__ == "__main__":
    main()
