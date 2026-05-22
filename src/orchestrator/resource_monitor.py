from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict


def resource_snapshot() -> Dict:
    disk = shutil.disk_usage(Path("."))
    gpu = "unknown"
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu = "apple_mps"
        elif torch.cuda.is_available():
            gpu = "cuda"
        else:
            gpu = "none"
    except Exception:
        gpu = "torch_unavailable"
    return {
        "python": sys.version.split()[0],
        "gpu": gpu,
        "disk_free_gb": round(disk.free / (1024**3), 3),
        "cpu_safe_mode": gpu in {"none", "torch_unavailable", "apple_mps"},
    }


def disk_ok(min_free_gb: float = 2.0) -> bool:
    return resource_snapshot()["disk_free_gb"] >= min_free_gb

