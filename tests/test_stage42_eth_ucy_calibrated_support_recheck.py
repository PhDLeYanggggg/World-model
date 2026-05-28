import numpy as np

from src import stage42_eth_ucy_calibrated_support_recheck as jm


def test_rel_to_source_id_maps_eth_ucy_sources():
    assert jm._rel_to_source_id("ETH/seq_eth/obsmat.txt") == "ETH_seq_eth"
    assert jm._rel_to_source_id("UCY/students03/obsmat.txt") == "UCY_students03"


def test_standardize_on_train_centers_training_rows():
    x = np.asarray([[1.0, 2.0], [3.0, 6.0], [10.0, 20.0]], dtype=np.float32)
    z = jm._standardize_on_train(x, np.asarray([True, True, False]))
    assert np.allclose(z[:2].mean(axis=0), 0.0, atol=1e-6)


def test_safe_requires_easy_guard():
    assert jm._safe(
        {
            "all_improvement": 0.01,
            "t50_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
        }
    )
    assert not jm._safe(
        {
            "all_improvement": 0.50,
            "t50_improvement": 0.50,
            "hard_failure_improvement": 0.50,
            "easy_degradation": 0.03,
        }
    )
