from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

os.environ.setdefault("WORLD_MODEL_TORCH_THREADS", "4")
os.environ.setdefault("WORLD_MODEL_TORCH_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if (
    sys.platform == "darwin"
    and platform.machine().lower() == "x86_64"
    and os.environ.get("WORLD_MODEL_ALLOW_RISKY_OPENMP") != "1"
):
    raise RuntimeError(
        "Refusing to import torch training under macOS x86_64/Rosetta because this runtime "
        "can trigger Intel OpenMP Can't open SHM hangs before Python can recover. "
        "Use .venv-pytorch/bin/python on arm64, or set WORLD_MODEL_ALLOW_RISKY_OPENMP=1 "
        "only if you accept the crash risk."
    )

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.m3w.dataset import BASELINE_NAMES, load_datasets
from src.m3w.models import M3WModel
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _int_setting(config: Dict[str, Any], key: str, env_key: str, default: int) -> int:
    raw = os.environ.get(env_key, config.get(key, default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def configure_torch_threads(config: Dict[str, Any]) -> Dict[str, int]:
    intra = _int_setting(config, "torch_threads", "WORLD_MODEL_TORCH_THREADS", 1)
    interop = _int_setting(config, "torch_interop_threads", "WORLD_MODEL_TORCH_INTEROP_THREADS", 1)
    torch.set_num_threads(intra)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        if torch.get_num_interop_threads() != interop:
            raise
    return {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}


def assert_safe_torch_runtime() -> None:
    """Refuse the known Apple Silicon Rosetta + Intel OpenMP crash path."""
    if os.environ.get("WORLD_MODEL_ALLOW_RISKY_OPENMP") == "1":
        return
    is_darwin_x86 = sys.platform == "darwin" and platform.machine().lower() == "x86_64"
    if is_darwin_x86 and torch.backends.mkl.is_available():
        raise RuntimeError(
            "Refusing M3W torch training in macOS x86_64/Rosetta + Intel MKL/OpenMP runtime. "
            "This environment has previously triggered OMP Can't open SHM crashes. "
            "Use .venv-pytorch/bin/python on arm64, or set WORLD_MODEL_ALLOW_RISKY_OPENMP=1 "
            "only if you intentionally accept the crash risk."
        )


def choose_device(config: Dict[str, Any], force_cpu: bool = False, force_mps: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if force_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.get("device") == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(dataset, batch_size: int, shuffle: bool, hard_oversample: bool = False) -> DataLoader:
    if hard_oversample:
        weights = 1.0 + 3.0 * dataset.hard_candidate + 4.0 * dataset.failure_label
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    per_sample_mse = ((outputs["log_fde"] - batch["y_log_fde"]) ** 2).mean(dim=1)
    sample_weight = 1.0 + float(weights.get("hard_weight", 0.0)) * batch["interaction"] + float(weights.get("failure_weight", 0.0)) * batch["failure"]
    mse = (per_sample_mse * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    if float(weights.get("ranking", 0.0)) > 0.0:
        pred_diff = outputs["log_fde"].unsqueeze(2) - outputs["log_fde"].unsqueeze(1)
        true_diff = batch["y_log_fde"].unsqueeze(2) - batch["y_log_fde"].unsqueeze(1)
        rank = nn.functional.mse_loss(torch.tanh(pred_diff), torch.tanh(true_diff))
    else:
        rank = torch.tensor(0.0, device=outputs["log_fde"].device)
    failure = nn.functional.binary_cross_entropy_with_logits(outputs["failure_logit"], batch["failure"])
    interaction = nn.functional.binary_cross_entropy_with_logits(outputs["interaction_logit"], batch["interaction"])
    occupancy = nn.functional.mse_loss(outputs["occupancy"], batch["occupancy"])
    return (
        float(weights.get("fde", 1.0)) * mse
        + float(weights.get("ranking", 0.0)) * rank
        + float(weights.get("failure", 0.4)) * failure
        + float(weights.get("interaction", 0.2)) * interaction
        + float(weights.get("occupancy", 0.1)) * occupancy
    )


def _jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_n = nn.functional.normalize(pred, dim=-1)
    target_n = nn.functional.normalize(target, dim=-1)
    cosine = 1.0 - (pred_n * target_n).sum(dim=-1).mean()
    l2 = nn.functional.mse_loss(pred, target)
    var = pred.float().var(dim=0).mean()
    collapse_penalty = torch.relu(torch.tensor(0.05, device=pred.device) - var)
    return l2 + cosine + collapse_penalty, {"jepa_l2": float(l2.detach().cpu()), "jepa_cosine": float(cosine.detach().cpu()), "latent_variance": float(var.detach().cpu())}


@torch.no_grad()
def _val_loss(model: M3WModel, loader: DataLoader, device: torch.device, weights: Dict[str, float]) -> float:
    model.eval()
    vals = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        vals.append(float(_loss(model(batch["x"]), batch, weights).detach().cpu()))
    return float(np.mean(vals)) if vals else 0.0


def train_m3w(
    config_path: str | Path,
    quick: bool = False,
    medium: bool = False,
    long: bool = False,
    resume: bool = False,
    mps: bool = False,
    cpu: bool = False,
    checkpoint_every: int | None = None,
    heartbeat_minutes: float | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    assert_safe_torch_runtime()
    thread_info = configure_torch_threads(config)
    if quick:
        config["jepa_epochs"] = min(int(config.get("jepa_epochs", 4)), 2)
        config["supervised_epochs"] = min(int(config.get("supervised_epochs", 8)), 3)
    if checkpoint_every is not None:
        config["checkpoint_every"] = checkpoint_every
    if heartbeat_minutes is not None:
        config["heartbeat_minutes"] = heartbeat_minutes
    _seed(int(config.get("seed", 27)))
    device = choose_device(config, force_cpu=cpu, force_mps=mps)
    out_dir = ensure_dir(config.get("output_dir", "outputs/m3w"))
    ckpt_dir = ensure_dir(Path(out_dir) / "checkpoints")
    train_ds, val_ds, _test_ds = load_datasets(config)
    train_loader = _loader(train_ds, int(config["batch_size"]), True, bool(config.get("hard_oversample", False)))
    val_loader = _loader(val_ds, int(config["batch_size"]), False)
    variants = ["jepa_only", "transformer_only", "hybrid"]
    results = {}
    best = {"variant": None, "val_loss": float("inf"), "checkpoint": None}
    best_name = "best_medium.pt" if str(config.get("mode", "")).startswith("medium") else "best_small.pt"
    start = time.time()
    _heartbeat(out_dir, "initializing local-small training", start, {})
    write_json(Path(out_dir) / "torch_runtime.json", thread_info)
    for variant in variants:
        model = M3WModel(train_ds.schema, config, variant=variant).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
        jepa_stats = []
        if variant in {"jepa_only", "hybrid"}:
            for epoch in range(int(config["jepa_epochs"])):
                model.train()
                losses = []
                for batch in train_loader:
                    x = batch["x"].to(device)
                    opt.zero_grad(set_to_none=True)
                    out = model.jepa_forward(x, float(config.get("mask_ratio", 0.35)))
                    loss, stats = _jepa_loss(out["pred"], out["target"])
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach().cpu()))
                jepa_stats.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **stats})
                _heartbeat(out_dir, f"pretrain {variant} epoch {epoch + 1}", start, results)
        weights = config.get("loss_weights", {})
        val_history = []
        best_variant_val = float("inf")
        best_variant_path = ckpt_dir / f"{variant}_best.pt"
        for epoch in range(int(config["supervised_epochs"])):
            model.train()
            losses = []
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                loss = _loss(model(batch["x"]), batch, weights)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            val = _val_loss(model, val_loader, device, weights)
            val_history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val})
            if val < best_variant_val:
                best_variant_val = val
                torch.save(_checkpoint(model, config, train_ds, variant, val, jepa_stats), best_variant_path)
            if (epoch + 1) % int(config.get("checkpoint_every", 2)) == 0:
                torch.save(_checkpoint(model, config, train_ds, variant, val, jepa_stats), ckpt_dir / f"{variant}_epoch{epoch + 1}.pt")
            _heartbeat(out_dir, f"supervised {variant} epoch {epoch + 1}", start, results)
        final_val = best_variant_val if val_history else float("inf")
        ckpt_path = ckpt_dir / f"{variant}_final.pt"
        torch.save(_checkpoint(model, config, train_ds, variant, final_val, jepa_stats), ckpt_path)
        results[variant] = {"val_loss": final_val, "jepa_stats": jepa_stats, "val_history": val_history, "checkpoint": str(best_variant_path)}
        if final_val < best["val_loss"]:
            best = {"variant": variant, "val_loss": final_val, "checkpoint": str(best_variant_path)}
            torch.save(torch.load(best_variant_path, map_location="cpu"), ckpt_dir / best_name)
    report = {
        "project_name": "M3W: Real-World Multimodal Agent-Scene World Model",
        "mode": config.get("mode", "small"),
        "backend": "torch_cpu_multithread" if str(device) == "cpu" and thread_info["torch_threads"] > 1 else f"torch_{device}",
        "device": str(device),
        "torch_runtime": thread_info,
        "variants": results,
        "best": best,
        "true_3d": False,
        "foundation_world_model": False,
        "latent_generative": False,
        "smc": False,
        "ordinary_residual_trained": False,
        "elapsed_s": time.time() - start,
        "resume_requested": bool(resume),
        "heartbeat_minutes": config.get("heartbeat_minutes"),
    }
    write_json(Path(out_dir) / "training_report.json", report)
    write_md(
        Path(out_dir) / "training_report.md",
        [
            "# M3W Training Report",
            "",
            "- This is a local-small representation/deterministic-head run, not a foundation/full model.",
            "- No Stage5C latent generative execution, no SMC, no ordinary residual training.",
            f"- device: `{device}`",
            f"- torch threads: `{thread_info['torch_threads']}`",
            f"- torch interop threads: `{thread_info['torch_interop_threads']}`",
            f"- best variant: `{best['variant']}`",
            f"- best checkpoint: `{best['checkpoint']}`",
        ],
    )
    return report


def _checkpoint(model: M3WModel, config: Dict[str, Any], dataset, variant: str, val_loss: float, jepa_stats: Iterable[Dict[str, float]]) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "config": config,
        "variant": variant,
        "val_loss": val_loss,
        "feature_mean": dataset.mean.astype(float).tolist(),
        "feature_std": dataset.std.astype(float).tolist(),
        "feature_names": dataset.feature_names,
        "token_to_features": dataset.schema.token_to_features,
        "baseline_names": BASELINE_NAMES,
        "jepa_stats": list(jepa_stats),
    }


def _heartbeat(out_dir: Path, task: str, start: float, results: Dict[str, Any]) -> None:
    write_md(
        Path(out_dir) / "heartbeat.md",
        [
            "# M3W Heartbeat",
            "",
            f"- current task: `{task}`",
            f"- elapsed seconds: `{time.time() - start:.1f}`",
            f"- completed variants: `{list(results.keys())}`",
            "- latent generative: `False`",
            "- SMC: `False`",
        ],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--heartbeat-minutes", type=float, default=None)
    args = parser.parse_args(argv)
    train_m3w(
        args.config,
        quick=args.quick,
        medium=args.medium,
        long=args.long,
        resume=args.resume,
        mps=args.mps,
        cpu=args.cpu,
        checkpoint_every=args.checkpoint_every,
        heartbeat_minutes=args.heartbeat_minutes,
    )
