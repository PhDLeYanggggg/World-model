import numpy as np

from scripts.run_m3w_moving_history_guard_transfer import fit


def test_transfer_envelope_uses_source_labels_only():
    groups = np.repeat(["a", "b", "c"], 2)
    def estimator(x, y, g, train, pred):
        assert not set(g[train]) & set(g[pred])
        assert (g[train] != "moving_transfer_target").all()
        np.testing.assert_array_equal(y[g == "moving_transfer_target"], 0.)
        return np.full(len(pred), y[train].mean())
    p, low, gap = fit(np.zeros((6, 1)), np.arange(6.), groups, np.zeros((3, 1)), estimator)
    np.testing.assert_array_equal(p, 2.5)
    assert gap > 0.
    np.testing.assert_allclose(low, p-gap)
