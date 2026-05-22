from __future__ import annotations

import numpy as np


class Stage18TargetEncoder:
    """Stop-gradient style target encoder placeholder for latent targets."""

    def encode(self, target: np.ndarray) -> np.ndarray:
        return np.asarray(target, dtype=np.float64).copy()

