import numpy as np

from src import stage41_domain_local_neural_retrain as dl


def test_seq_summary_returns_mean_last_span_and_valid_fraction():
    seq = np.zeros((2, 4, 7), dtype=np.float32)
    seq[0, :, 0] = [1, 2, 3, 4]
    seq[0, :, -1] = [1, 1, 1, 1]
    seq[1, :, 0] = [10, 20, 30, 40]
    seq[1, :, -1] = [1, 1, 0, 0]
    out = dl._seq_summary(seq)
    assert out.shape == (2, 19)
    assert out[0, 0] == 2.5
    assert out[0, 6] == 4
    assert out[0, -1] == 1.0
    assert out[1, 0] == 15.0
    assert out[1, 6] == 20
    assert out[1, -1] == 0.5


def test_domain_mask_uses_dataset_domain_field():
    data = {"domain": np.asarray(["ETH_UCY", "UCY", "TrajNet", "UCY"])}
    assert dl._domain_mask(data, "UCY").tolist() == [False, True, False, True]


def test_ridge_fit_predicts_linear_gain():
    x = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, -1.0], [3.0, 0.5]], dtype=np.float32)
    y = 1.5 * x[:, 0] - 0.25 * x[:, 1] + 2.0
    w = dl._ridge_fit(x, y, lam=1e-8)
    pred = dl._ridge_predict(x, w)
    assert np.max(np.abs(pred - y)) < 1e-5
