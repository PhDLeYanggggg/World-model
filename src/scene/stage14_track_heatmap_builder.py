from __future__ import annotations

from typing import Dict


def heatmap_status() -> Dict[str, str]:
    return {
        "endpoint_heatmap_source": "train_split_only",
        "test_endpoints_used": "false",
    }

