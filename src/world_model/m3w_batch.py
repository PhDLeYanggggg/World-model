from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class M3WBatch:
    features: np.ndarray
    y_fde: np.ndarray
    horizon: np.ndarray
    split_type: np.ndarray
    strongest_idx: np.ndarray
    hard_candidate: np.ndarray
    data_role: str
    token_valid_mask: Dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return int(len(self.features))
