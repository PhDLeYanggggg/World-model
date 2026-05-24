from pathlib import Path

import numpy as np

from src import stage41_stratified_protocol as strat


def test_stage41_stratified_paths() -> None:
    assert strat.OUT_DIR == Path("outputs/stage41_stratified_protocol")
    assert strat.DATA_DIR == Path("data/stage41_stratified_protocol")


def test_stage41_stratified_split_mask_shape() -> None:
    idx = strat._candidate_split_index()
    n = int(idx["row_id"].max()) + 1
    mask = strat._split_mask("train", n)
    assert mask.dtype == bool
    assert mask.shape == (n,)
    assert np.any(mask)
