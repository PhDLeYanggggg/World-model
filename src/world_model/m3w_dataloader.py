from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from src.m3w.dataset import DATA_ROLES, M3WFeatureDataset
from src.m3w.token_schema import TOKEN_NAMES
from src.world_model.m3w_batch import M3WBatch


class M3WDataLoader:
    """Single-process M3W dataloader with explicit data roles.

    This loader wraps the Stage26 SDD feature store and never uses DataLoader
    multiprocessing. It is intended for audits, evidence generation, and
    CPU-safe research loops.
    """

    def __init__(self, feature_store: str | Path, split: str, batch_size: int = 2048, limit: int | None = None) -> None:
        self.dataset = M3WFeatureDataset(feature_store, split, limit=limit)
        self.batch_size = int(batch_size)
        self.data_role = DATA_ROLES.get(split, "diagnostic_only")

    def __iter__(self) -> Iterator[M3WBatch]:
        for start in range(0, len(self.dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dataset))
            idx = slice(start, end)
            valid = {token: np.ones(end - start, dtype=bool) for token in TOKEN_NAMES}
            yield M3WBatch(
                features=self.dataset.x[idx],
                y_fde=self.dataset.y_fde[idx],
                horizon=self.dataset.horizon[idx],
                split_type=self.dataset.split_type[idx],
                strongest_idx=self.dataset.strongest_idx[idx],
                hard_candidate=self.dataset.hard_candidate[idx],
                data_role=self.data_role,
                token_valid_mask=valid,
            )
