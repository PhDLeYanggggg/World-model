import numpy as np
from scripts.audit_m3w_cost_transfer_features import feature_names
from scripts.run_m3w_bounded_cost import features


def test_feature_names_follow_actual_past_and_rollout_schema():
    names = feature_names()
    assert len(names) == len(set(names)) == 356
    assert names[331] == 'candidate_rollout_0_y'
    assert names[354:] == ['log_past_scale', 'log1p_native_disagreement']
    g = np.zeros((1, 476), np.float32)
    g[:, 16:24] = np.arange(-7, 1)/12
    b = np.column_stack((np.arange(1, 13)/12, np.zeros(12))).astype(np.float32)
    g[:, 332:356] = b.ravel()
    p = b[None].copy(); p[:, :, 1] = .2
    x, d, same = features(g, p, np.array([3.]))
    assert not same[0]
    np.testing.assert_allclose(x[0, 288:294], [b[:, 0].mean(), 0, b[:, 0].std(), 0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(x[0, 294:300], [b[:, 0].mean(), .2, b[:, 0].std(), 0, 1, .2], atol=1e-7)
    np.testing.assert_array_equal(x[0, 306:330], b.ravel())
    np.testing.assert_array_equal(x[0, 330:354], p.ravel())
    np.testing.assert_allclose(x[0, 354:], [np.log(3), np.log1p(d[0])], atol=1e-7)
