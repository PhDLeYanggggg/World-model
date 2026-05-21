from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from src.models.encoders.stage5b6_interaction_encoder import Stage5B6InteractionEncoder


class Stage6InteractionEncoder:
    """Causal interaction encoder with explicit ablation modes."""

    def __init__(self, dataset: str, root: str | Path = "data/stage5b_world_state", mode: str = "graph_temporal"):
        self.dataset = dataset
        self.mode = mode
        self.encoder = Stage5B6InteractionEncoder(dataset, root=root)

    def encode(self, meta: Dict) -> np.ndarray:
        full = self.encoder.encode_episode(meta).as_array()
        if self.mode == "none":
            return np.zeros_like(full)
        if self.mode == "scalar":
            out = np.zeros_like(full)
            out[:4] = full[:4]
            return out
        if self.mode == "graph_single":
            out = np.zeros_like(full)
            out[:7] = full[:7]
            return out
        return full

