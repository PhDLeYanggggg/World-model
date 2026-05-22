from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple


@dataclass
class Stage17BaselineSelector:
    """Conservative causal rule selector for Stage 17 baseline choices."""

    rules: Sequence[Dict[str, Any]]
    fallback_baseline: str = "constant_position"

    def select(self, features: Dict[str, float]) -> Tuple[str, float]:
        for rule in self.rules:
            value = float(features.get(rule["feature"], 0.0) or 0.0)
            threshold = float(rule["threshold"])
            matched = value > threshold if rule["op"] == ">" else value < threshold
            if matched:
                return str(rule["baseline_id"]), float(rule.get("confidence", 0.65))
        return self.fallback_baseline, 0.78

