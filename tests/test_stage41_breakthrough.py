from pathlib import Path

import numpy as np

from src import stage41_breakthrough as s41


def test_stage41_paths_and_candidate_schema() -> None:
    assert s41.OUT_DIR == Path("outputs/stage41_breakthrough")
    assert s41.SPLIT_OUT == Path("outputs/stage41_external_split")
    assert s41.DATA_DIR == Path("data/stage41_world_model")
    assert s41.CANDIDATE_NAMES[0] == "stage41_train_horizon_strongest"
    assert len(s41.CANDIDATE_NAMES) >= 2


def test_stage41_rebuilt_split_has_no_group_overlap() -> None:
    report = s41.rebuild_external_split()
    idx = dict(np.load(s41.DATA_DIR / "stage41_split_index.npz"))
    split = idx["split"].astype(str)
    group = idx["group"].astype(str)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(group[split == a].tolist()).isdisjoint(set(group[split == b].tolist()))
    assert report["no_leakage"]["future_endpoint_input"] is False
    assert report["no_leakage"]["central_velocity"] is False


def test_stage41_metrics_fallback_zero_improvement() -> None:
    s41.build_seq2seq_dataset()
    ds = s41._ds("test")
    fallback = ds["floor_fde"].astype(float)
    m = s41._metrics(fallback.copy(), fallback, ds)
    assert m["all_improvement"] == 0.0
    assert m["easy_degradation"] == 0.0
