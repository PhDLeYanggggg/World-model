import numpy as np

from src import stage41_dynamic_shape_meta_policy as meta


def test_dynamic_source_distribution_sums_to_one():
    chosen = np.asarray([0, 1, 2, 2], dtype=np.int64)
    dist = meta._source_distribution(chosen)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["gain_gate"] == 0.5


def test_dynamic_policy_falls_back_when_gain_too_small():
    pack = {
        "horizon": np.asarray([50, 100], dtype=np.int64),
        "bridge_xy": np.zeros((2, 4, 2), dtype=np.float64),
        "old_shape": {"xy": np.ones((2, 4, 2), dtype=np.float64), "shape_switch": np.asarray([True, True])},
        "gain_gate": {"xy": np.ones((2, 4, 2), dtype=np.float64) * 2.0, "shape_switch": np.asarray([True, True])},
    }
    pred = {
        "bridge": np.asarray([1.0, 1.0], dtype=np.float64),
        "old_shape": np.asarray([0.99, 0.99], dtype=np.float64),
        "gain_gate": np.asarray([0.98, 0.98], dtype=np.float64),
    }
    policy = {"gain_min": 0.1, "margin_min": 0.0, "max_rate_h10": 0.0, "max_rate_h25": 0.0, "max_rate_h50": 1.0, "max_rate_h100": 1.0}
    xy, switch, chosen = meta._choose_dynamic_sources(pack, pred, policy)
    assert np.all(chosen == 0)
    assert not np.any(switch)
    assert np.allclose(xy, pack["bridge_xy"])


def test_prediction_costs_are_finite_after_log_clipping():
    pack = {
        "horizon": np.asarray([50], dtype=np.int16),
        "data": {
            "horizon": np.asarray([50], dtype=np.int16),
            "normalizer": np.asarray([1.0], dtype=np.float32),
            "current_xy": np.asarray([[0.0, 0.0]], dtype=np.float32),
            "seq": np.zeros((1, 64, 7), dtype=np.float32),
            "static": np.zeros((1, 82), dtype=np.float32),
            "cand_delta": np.zeros((1, 9, 2), dtype=np.float32),
        },
        "endpoint_pred": {"delta": np.zeros((1, 2), dtype=np.float32), "uncertainty": np.zeros(1, dtype=np.float32)},
        "bridge_xy": np.zeros((1, 4, 2), dtype=np.float64),
        "old_shape": {"xy": np.zeros((1, 4, 2), dtype=np.float64), "shape_switch": np.zeros(1, dtype=bool)},
        "gain_gate": {"xy": np.zeros((1, 4, 2), dtype=np.float64), "shape_switch": np.zeros(1, dtype=bool)},
    }
    dim = meta._source_feature_matrix(pack, "bridge").shape[1]
    model = {"mean": np.zeros(dim), "std": np.ones(dim), "w": np.ones(dim + 1) * 1e6}
    pred = meta._predict_source_costs(model, pack)
    assert np.isfinite(pred["bridge"]).all()
