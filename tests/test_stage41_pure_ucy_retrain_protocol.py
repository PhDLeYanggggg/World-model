import numpy as np

from src import stage41_pure_ucy_retrain_protocol as proto


def test_ridge_fit_recovers_simple_linear_signal():
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5
    w = proto._ridge_fit(x, y, lam=1e-8)
    pred = proto._ridge_predict(x, w)
    assert np.max(np.abs(pred - y)) < 1e-5


def test_pure_ucy_mask_uses_official_ucy_source_prefix():
    data = {
        "labels": {
            "source_file": np.asarray(
                [
                    "/root/datasets/UCY/zara01/obsmat.txt",
                    "/root/datasets/TrajNet/Train/crowds/crowds_zara03.txt",
                    "/root/datasets/UCY/zara02/obsmat.txt",
                ]
            ),
            "domain": np.asarray(["UCY", "UCY", "TrajNet"]),
        }
    }
    assert proto._pure_ucy_mask(data).tolist() == [True, False, True]


def test_alpha_policy_is_fallback_safe_when_gain_below_threshold():
    data = {
        "proposal_harm": np.asarray([0.1, 0.1], dtype=np.float32),
        "proposal_uncertainty": np.asarray([0.1, 0.1], dtype=np.float32),
        "proposal_teacher_prob": np.asarray([0.9, 0.9], dtype=np.float32),
        "teacher_repaired_switch": np.asarray([True, True]),
        "teacher_raw_switch": np.asarray([True, True]),
    }
    pred = {"pred_gain": np.asarray([0.2, -0.1]), "pred_harm": np.asarray([0.0, 0.0])}
    policy = {"mode": "gain_harm", "gain_min": 0.0, "harm_max": 0.1, "proposal_harm_max": 0.2, "uncertainty_max": 0.2, "alpha": 0.4}
    alpha = proto._alpha_for_policy(data, pred, policy)
    assert alpha.tolist() == [0.4, 0.0]
