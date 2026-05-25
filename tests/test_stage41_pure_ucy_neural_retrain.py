import numpy as np

from src import stage41_pure_ucy_neural_retrain as pure


def test_pure_ucy_split_uses_only_official_ucy_sources():
    assert pure._pure_ucy_split("/tmp/datasets/UCY/students01/students001-trajnet.txt") == "train"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/students03/obsmat.txt") == "train"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/zara01/obsmat.txt") == "val"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/zara02/obsmat.txt") == "test"
    assert pure._pure_ucy_split("/tmp/datasets/UCY/zara03/crowds_zara03.txt") == "test"
    assert pure._pure_ucy_split("/tmp/datasets/TrajNet/Train/crowds/crowds_zara03.txt") == "unused"


def test_select_policy_falls_back_when_switch_budget_zero():
    pred = {
        "candidate_score": np.asarray([[1.0, 0.1], [1.0, 0.2]], dtype=np.float32),
        "gain": np.asarray([1.0, 1.0], dtype=np.float32),
        "harm": np.asarray([0.0, 0.0], dtype=np.float32),
        "physical": np.asarray([1.0, 1.0], dtype=np.float32),
    }
    ds = {
        "candidate_fde": np.asarray([[10.0, 2.0], [20.0, 5.0]], dtype=np.float32),
        "horizon": np.asarray([50, 50], dtype=np.int16),
        "hard": np.asarray([True, False]),
        "failure": np.asarray([False, False]),
    }
    selected, switch, selected_idx = pure._select({"max_switch": 0.0}, pred, ds)
    assert selected.tolist() == [10.0, 20.0]
    assert switch.tolist() == [False, False]
    assert selected_idx.tolist() == [0, 0]


def test_strict_gate_requires_positive_switch_and_easy_preservation():
    assert pure._strict_gate(
        {
            "all_improvement": 0.01,
            "t50_improvement": 0.01,
            "hard_failure_improvement": 0.01,
            "easy_degradation": 0.02,
            "switch_rate": 0.01,
        }
    )
    assert not pure._strict_gate(
        {
            "all_improvement": 0.01,
            "t50_improvement": 0.01,
            "hard_failure_improvement": 0.01,
            "easy_degradation": 0.021,
            "switch_rate": 0.01,
        }
    )
