from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.synthetic_dataset import split_episodes
from src.models.neural_residual_world_model import (
    DeterministicResidualWorldModel,
    apply_normalization,
    build_feature_tensors,
    fit_normalization,
)
from src.models.stochastic_world_model import StochasticResidualWorldModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train_stage2_models(episodes: list[Dict], cfg: Dict, quick: bool = True) -> Dict:
    seed_everything(int(cfg["seed"]))
    train_episodes = split_episodes(episodes, "train")
    val_episodes = split_episodes(episodes, "val")
    history_steps = int(cfg["training"]["history_steps"])
    train_raw = build_feature_tensors(train_episodes, history_steps)
    val_raw = build_feature_tensors(val_episodes, history_steps)
    normalization = fit_normalization(train_raw)
    train_pack = apply_normalization(train_raw, normalization)
    val_pack = apply_normalization(val_raw, normalization)

    deterministic, deterministic_log = train_one_model(
        train_pack,
        val_pack,
        cfg,
        stochastic=False,
        quick=quick,
    )
    stochastic, stochastic_log = train_one_model(
        train_pack,
        val_pack,
        cfg,
        stochastic=True,
        quick=quick,
    )

    model_dir = Path(cfg["output_dir"]) / "models" / "stage2"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(deterministic.state_dict(), model_dir / "deterministic_neural_residual.pt")
    torch.save(stochastic.state_dict(), model_dir / "stochastic_neural_residual.pt")
    serializable_norm = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in normalization.items()}
    (model_dir / "normalization.json").write_text(json.dumps(serializable_norm, indent=2), encoding="utf-8")
    return {
        "deterministic_model": deterministic,
        "stochastic_model": stochastic,
        "normalization": normalization,
        "training": {
            "deterministic_neural_residual": deterministic_log,
            "stochastic_neural_residual": stochastic_log,
            "model_dir": str(model_dir),
            "history_steps": history_steps,
            "target": "learned residual acceleration: true_next_acceleration - hand_physics_acceleration",
        },
    }


def train_one_model(train_pack: Dict, val_pack: Dict, cfg: Dict, stochastic: bool, quick: bool) -> Tuple[nn.Module, Dict]:
    device = torch.device(cfg.get("device", "cpu"))
    model_cls = StochasticResidualWorldModel if stochastic else DeterministicResidualWorldModel
    model = model_cls(
        entity_dim=train_pack["entity"].shape[-1],
        neighbor_dim=train_pack["neighbor"].shape[-1],
        obstacle_dim=train_pack["obstacle"].shape[-1],
    ).to(device)
    loader = DataLoader(make_dataset(train_pack), batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_tensors = tensors_for_pack(val_pack, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["quick_epochs" if quick else "epochs"])
    best_state = None
    best_val = float("inf")
    history = []
    target_mean = torch.tensor(train_pack["target_raw"].mean(axis=0), dtype=torch.float32, device=device)
    target_std = torch.tensor(train_pack["target_raw"].std(axis=0) + 1e-6, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            tensors = tuple(item.to(device) for item in batch)
            loss, components = model_loss(model, tensors, target_mean, target_std, cfg, stochastic=stochastic)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            epoch_losses.append(components)

        model.eval()
        with torch.no_grad():
            val_loss, val_components = model_loss(model, val_tensors, target_mean, target_std, cfg, stochastic=False)
        train_components = average_components(epoch_losses)
        row = {
            "epoch": int(epoch),
            "train_total_loss": round(float(train_components["total_loss"]), 6),
            "val_total_loss": round(float(val_loss.item()), 6),
            **{f"train_{key}": round(float(value), 6) for key, value in train_components.items() if key != "total_loss"},
            **{f"val_{key}": round(float(value), 6) for key, value in val_components.items() if key != "total_loss"},
        }
        history.append(row)
        if float(val_loss.item()) < best_val:
            best_val = float(val_loss.item())
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "model": "stochastic_neural_residual" if stochastic else "deterministic_neural_residual",
        "train_samples": int(train_pack["entity"].shape[0]),
        "val_samples": int(val_pack["entity"].shape[0]),
        "epochs": epochs,
        "best_val_total_loss": round(best_val, 6),
        "loss_components": history,
        "notes": "collision/obstacle/boundary terms are local differentiable surrogates; rollout evaluation reports true geometric violations.",
    }


def make_dataset(pack: Dict) -> TensorDataset:
    return TensorDataset(*[torch.tensor(pack[key], dtype=torch.float32) for key in tensor_keys()])


def tensors_for_pack(pack: Dict, device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(pack[key], dtype=torch.float32, device=device) for key in tensor_keys())


def tensor_keys() -> list[str]:
    return ["entity", "neighbor", "obstacle", "target", "current", "next", "base_accel", "scene_bounds", "neighbor_raw", "obstacle_raw"]


def model_loss(
    model: nn.Module,
    tensors: Tuple[torch.Tensor, ...],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    cfg: Dict,
    stochastic: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    entity, neighbor, obstacle, target, current, next_state, base_accel, scene_bounds, neighbor_raw, obstacle_raw = tensors
    latent = None
    diversity_loss = torch.tensor(0.0, dtype=torch.float32, device=entity.device)
    kl_loss = torch.tensor(0.0, dtype=torch.float32, device=entity.device)
    if getattr(model, "latent_dim", 0):
        latent = torch.randn((entity.shape[0], int(model.latent_dim)), dtype=torch.float32, device=entity.device) * (0.45 if stochastic else 0.0)
        kl_loss = 0.5 * torch.mean(latent * latent) * 0.001
    pred, log_std = model(entity, neighbor, obstacle, latent)
    dynamics_loss = nn.functional.smooth_l1_loss(pred, target)
    nll_loss = torch.mean(0.5 * ((pred - target) / torch.exp(log_std)) ** 2 + log_std)

    pred_residual = pred * target_std + target_mean
    true_residual = target * target_std + target_mean
    pred_accel = base_accel + pred_residual
    true_accel = base_accel + true_residual
    max_accel = current[:, 9:10]
    pred_accel = clip_torch_vectors(pred_accel, max_accel)
    pred_velocity = clip_torch_vectors(current[:, 2:4] + pred_accel * float(cfg["world"]["dt"]), current[:, 8:9])
    true_velocity = next_state[:, 2:4]
    pred_position = current[:, :2] + pred_velocity * float(cfg["world"]["dt"])
    true_position = next_state[:, :2]

    position_loss = nn.functional.mse_loss(pred_position, true_position)
    velocity_loss = nn.functional.mse_loss(pred_velocity, true_velocity)
    acceleration_loss = nn.functional.mse_loss(pred_accel, true_accel)
    heading_pred = torch.atan2(pred_velocity[:, 1], pred_velocity[:, 0])
    heading_loss = torch.mean(1.0 - torch.cos(heading_pred - next_state[:, 6]))
    goal_dir_loss = goal_direction_loss_fn(pred_position, true_position, current[:, 10:12])
    collision_penalty = torch.mean(torch.relu(0.05 - neighbor_raw[:, 6]) ** 2)
    obstacle_violation_penalty = torch.mean(torch.relu(0.10 - obstacle_raw[:, 2]) ** 2 + obstacle_raw[:, 3] ** 2)
    width, height = scene_bounds[:, 0], scene_bounds[:, 1]
    radius = current[:, 7]
    boundary_violation_penalty = torch.mean(
        torch.relu(radius - pred_position[:, 0])
        + torch.relu(pred_position[:, 0] - (width - radius))
        + torch.relu(radius - pred_position[:, 1])
        + torch.relu(pred_position[:, 1] - (height - radius))
    )
    max_speed_penalty = torch.mean(torch.relu(torch.linalg.norm(pred_velocity, dim=1) - current[:, 8]) ** 2)
    max_acceleration_penalty = torch.mean(torch.relu(torch.linalg.norm(pred_accel, dim=1) - current[:, 9]) ** 2)
    smoothness_penalty = torch.mean((pred_accel - current[:, 4:6]) ** 2)
    if stochastic and getattr(model, "latent_dim", 0):
        latent2 = torch.randn_like(latent) * 0.45
        pred2, _ = model(entity, neighbor, obstacle, latent2)
        diversity_loss = -0.002 * torch.mean(torch.abs(pred2 - pred))

    total = (
        dynamics_loss
        + 0.15 * nll_loss
        + 0.05 * position_loss
        + 0.03 * velocity_loss
        + 0.08 * acceleration_loss
        + 0.02 * heading_loss
        + 0.02 * goal_dir_loss
        + 0.04 * collision_penalty
        + 0.04 * obstacle_violation_penalty
        + 0.04 * boundary_violation_penalty
        + 0.02 * max_speed_penalty
        + 0.02 * max_acceleration_penalty
        + 0.01 * smoothness_penalty
        + diversity_loss
        + kl_loss
    )
    components = {
        "total_loss": float(total.detach().cpu()),
        "position_loss": float(position_loss.detach().cpu()),
        "velocity_loss": float(velocity_loss.detach().cpu()),
        "acceleration_loss": float(acceleration_loss.detach().cpu()),
        "heading_loss": float(heading_loss.detach().cpu()),
        "goal_direction_loss": float(goal_dir_loss.detach().cpu()),
        "collision_penalty": float(collision_penalty.detach().cpu()),
        "obstacle_violation_penalty": float(obstacle_violation_penalty.detach().cpu()),
        "boundary_violation_penalty": float(boundary_violation_penalty.detach().cpu()),
        "max_speed_penalty": float(max_speed_penalty.detach().cpu()),
        "max_acceleration_penalty": float(max_acceleration_penalty.detach().cpu()),
        "smoothness_penalty": float(smoothness_penalty.detach().cpu()),
        "stochastic_diversity_loss": float(diversity_loss.detach().cpu()),
        "KL_loss": float(kl_loss.detach().cpu()),
        "scaled_residual_loss": float(dynamics_loss.detach().cpu()),
        "uncertainty_nll_loss": float(nll_loss.detach().cpu()),
    }
    return total, components


def clip_torch_vectors(vectors: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norms), max_norm / torch.clamp(norms, min=1e-6))
    return vectors * scale


def goal_direction_loss_fn(pred_position: torch.Tensor, true_position: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    pred_dir = goals - pred_position
    true_dir = goals - true_position
    pred_dir = pred_dir / torch.clamp(torch.linalg.norm(pred_dir, dim=1, keepdim=True), min=1e-6)
    true_dir = true_dir / torch.clamp(torch.linalg.norm(true_dir, dim=1, keepdim=True), min=1e-6)
    return torch.mean((pred_dir - true_dir) ** 2)


def average_components(rows: list[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
