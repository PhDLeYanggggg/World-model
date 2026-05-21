from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.models.stage5b5_temporal_interaction_model import Stage5B5TemporalInteractionModel


DATASET_IDS = {"eth_ucy": 0, "tgsim": 1, "tgsim_i90": 2, "trajnet": 3}


def load_baseline_payload():
    return json.loads(Path("outputs/reports/stage5b_baseline_metrics.json").read_text(encoding="utf-8"))


def load_hard_lookup() -> Dict[Tuple[str, int], Dict]:
    path = Path("outputs/reports/stage5b5_hard_subset_summary.json")
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for dataset in rows:
        for ep in dataset.get("episodes", []):
            lookup[(dataset["dataset_name"], int(ep["episode_id"]))] = ep
    return lookup


def make_examples(mode: str = "hybrid") -> Tuple[List[Dict], Dict]:
    baselines = load_baseline_payload()
    hard = load_hard_lookup()
    examples = []
    for dataset, row in baselines["datasets"].items():
        baseline = row["strongest_causal_baseline"]
        for ep in load_dataset_episodes(dataset, split="train"):
            states = ep["states"]
            meta = ep["meta"]
            past = int(meta.get("past_horizon", 10))
            future = states.shape[0] - past
            dt = float(meta.get("dt_s", 1.0))
            base = rollout(states[:past], future, dt, baseline)[1:]
            true = states[past:]
            horizons = [1] if mode == "recurrent_rollout" else [h for h in [1, 10, 25, 50, 100] if h <= future]
            if mode == "hybrid":
                horizons = sorted(set(horizons + [min(5, future)]))
            hard_row = hard.get((dataset, int(meta.get("episode_id", -1))), {})
            weight = 2.0 if hard_row.get("hardness") == "hard" else (1.35 if hard_row.get("hardness") == "medium" else 1.0)
            neighbor = np.asarray(
                [
                    min(hard_row.get("nearest_neighbor_distance_min", 999.0), 50.0) / 50.0,
                    min(hard_row.get("time_to_collision_min", 999.0), 50.0) / 50.0,
                    hard_row.get("interaction_density", 0.0),
                    min(hard_row.get("closing_speed", 0.0), 20.0) / 20.0,
                ],
                dtype=np.float32,
            )
            scene = np.asarray([1.0 if dataset.startswith("tgsim") else 0.0, 1.0 if dataset in {"trajnet", "eth_ucy"} else 0.0, 0.0], dtype=np.float32)
            for h in horizons:
                examples.append(
                    {
                        "dataset": dataset,
                        "history": states[:past, 0, :].astype(np.float32),
                        "neighbor": neighbor,
                        "scene": scene,
                        "dataset_id": DATASET_IDS.get(dataset, 0),
                        "horizon": h,
                        "horizon_frac": h / max(future, 1),
                        "target": (true[h - 1, 0, 0:2] - base[h - 1, 0, 0:2]).astype(np.float32),
                        "weight": weight,
                    }
                )
    return examples, baselines


def train_temporal(mode: str = "hybrid", epochs: int = 120, seed: int = 7) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    examples, baselines = make_examples(mode)
    model = Stage5B5TemporalInteractionModel(num_datasets=8, residual_clip=4.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    if not examples:
        raise RuntimeError("No training examples. Run Stage 5B build, baseline, and hard mining first.")
    hist = torch.tensor(np.stack([e["history"] for e in examples]), dtype=torch.float32)
    neigh = torch.tensor(np.stack([e["neighbor"] for e in examples]), dtype=torch.float32)
    scene = torch.tensor(np.stack([e["scene"] for e in examples]), dtype=torch.float32)
    ds = torch.tensor([e["dataset_id"] for e in examples], dtype=torch.long)
    hfrac = torch.tensor([e["horizon_frac"] for e in examples], dtype=torch.float32)
    target = torch.tensor(np.stack([e["target"] for e in examples]), dtype=torch.float32)
    weight = torch.tensor([e["weight"] for e in examples], dtype=torch.float32)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred, gate = model(hist, neigh, scene, ds, hfrac)
        loss = (loss_fn(pred, target).mean(dim=1) * weight).mean()
        loss = loss + 0.002 * (pred.pow(2).mean()) + 0.001 * gate.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    out_dir = Path("outputs/checkpoints/stage5b5")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"temporal_interaction_{mode}.pt"
    torch.save({"state_dict": model.state_dict(), "dataset_ids": DATASET_IDS, "mode": mode}, ckpt)
    return {"mode": mode, "checkpoint": str(ckpt), "loss_final": round(losses[-1], 6), "loss_curve": losses}
