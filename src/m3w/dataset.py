from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.stage14_pipeline import read_json
from src.m3w.token_schema import TokenSchema, build_token_schema


BASELINE_NAMES = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity",
    "constant_acceleration_causal",
    "constant_turn_rate_velocity",
    "scene_clamped_baseline",
    "goal_directed_baseline",
]


DATA_ROLES = {
    "train": "supervised_training",
    "val": "supervised_training",
    "test": "official_eval",
}


class M3WFeatureDataset(Dataset):
    def __init__(
        self,
        feature_store: str | Path,
        split: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        limit: int | None = None,
    ) -> None:
        self.feature_store = Path(feature_store)
        self.split = split
        data = np.load(self.feature_store / f"{split}.npz")
        n = len(data["x"]) if limit is None else min(limit, len(data["x"]))
        self.x_raw = data["x"][:n].astype(np.float32)
        self.y_fde = data["y_fde"][:n].astype(np.float32)
        self.horizon = data["horizon"][:n]
        self.split_type = data["split_type"][:n]
        self.strongest_idx = data["strongest_idx"][:n].astype(np.int64)
        self.oracle_idx = data["oracle_idx"][:n].astype(np.int64)
        self.hard_candidate = data["hard_candidate"][:n].astype(np.float32)
        self.mean = mean if mean is not None else self.x_raw.mean(axis=0)
        self.std = std if std is not None else self.x_raw.std(axis=0) + 1e-6
        self.x = ((self.x_raw - self.mean) / self.std).astype(np.float32)
        manifest = read_json(self.feature_store / "manifest.json", {})
        self.feature_names = manifest.get("feature_names", [f"f{i}" for i in range(self.x.shape[1])])
        self.schema: TokenSchema = build_token_schema(self.feature_names)
        strong_err = self.y_fde[np.arange(n), self.strongest_idx]
        threshold = float(np.percentile(strong_err, 90)) if split == "train" else 95.20766448974608
        self.failure_label = (strong_err >= threshold).astype(np.float32)
        density_idx = self.feature_names.index("density_r50") if "density_r50" in self.feature_names else 0
        self.occupancy_target = np.clip(self.x_raw[:, density_idx] / 10.0, 0.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y_log_fde": torch.log1p(torch.from_numpy(self.y_fde[idx])),
            "y_fde": torch.from_numpy(self.y_fde[idx]),
            "strongest_idx": torch.tensor(self.strongest_idx[idx], dtype=torch.long),
            "oracle_idx": torch.tensor(self.oracle_idx[idx], dtype=torch.long),
            "failure": torch.tensor(self.failure_label[idx], dtype=torch.float32),
            "interaction": torch.tensor(self.hard_candidate[idx], dtype=torch.float32),
            "occupancy": torch.tensor(self.occupancy_target[idx], dtype=torch.float32),
            "horizon": torch.tensor(int(self.horizon[idx]), dtype=torch.long),
        }


def load_datasets(config: Dict[str, Any]) -> Tuple[M3WFeatureDataset, M3WFeatureDataset, M3WFeatureDataset]:
    store = config["feature_store"]
    train = M3WFeatureDataset(store, "train", limit=config.get("max_train_samples"))
    val = M3WFeatureDataset(store, "val", mean=train.mean, std=train.std, limit=config.get("max_val_samples"))
    test = M3WFeatureDataset(store, "test", mean=train.mean, std=train.std, limit=config.get("max_test_samples"))
    return train, val, test
