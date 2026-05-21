from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.models.stage5b6_gated_residual_model import sigmoid


class Stage7InteractionAuxiliary:
    def __init__(self, payload: Dict):
        self.payload = payload

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def predict(self, x: np.ndarray) -> Dict:
        mean = np.asarray(self.payload["x_mean"], dtype=float)
        scale = np.asarray(self.payload["x_scale"], dtype=float)
        xb = np.concatenate([[1.0], (np.asarray(x, dtype=float) - mean) / np.maximum(scale, 1e-6)])
        outputs = {}
        for name, coef in self.payload.get("heads", {}).items():
            val = float(xb @ np.asarray(coef, dtype=float))
            outputs[name] = float(sigmoid(val)) if name.endswith("_prob") else val
        return outputs

