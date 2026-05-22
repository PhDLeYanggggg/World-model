from __future__ import annotations

import numpy as np


def raster_summary(raster: np.ndarray) -> dict:
    return {"shape": list(raster.shape), "mean": float(np.mean(raster))}

