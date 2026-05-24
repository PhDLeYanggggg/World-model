from pathlib import Path

import numpy as np

from src import stage32_domain_alignment as s32


def test_stage32_output_paths_are_isolated() -> None:
    assert s32.OUT_DIR == Path("outputs/stage32_domain_alignment")
    assert s32.DATA_DIR == Path("data/stage32_domain_alignment")
    assert "velocity_scale" in s32.NORMALIZATIONS


def test_stage32_coral_transform_shape() -> None:
    x = np.eye(3, dtype=float)
    y = np.eye(3, dtype=float) * 2.0
    out = s32._coral_transform(x, x, y)
    assert out.shape == x.shape
    assert np.isfinite(out).all()
