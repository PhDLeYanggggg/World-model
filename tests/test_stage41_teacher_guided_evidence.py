import numpy as np

from src import stage41_teacher_guided_evidence as evidence


def test_evidence_pass_requires_positive_ci_and_domains() -> None:
    metrics = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "by_domain": {"A": {"all_improvement": 0.1}, "B": {"t50_improvement": 0.1}},
    }
    ci = {"all": {"low": 0.01}, "t50": {"low": 0.01}, "hard_failure": {"low": 0.01}}
    assert evidence._evidence_passes(metrics, ci, collision_delta=0.0)


def test_mask_data_zeros_named_slice(monkeypatch) -> None:
    monkeypatch.setattr(evidence.ft, "_fresh_ds", lambda _split: {"static": np.zeros((2, 3), dtype=np.float32)})
    data = {"x_teacher": np.ones((2, 39), dtype=np.float32)}
    masked = evidence._mask_data(data, ["prediction_signals"])
    assert np.all(masked["x_teacher"][:, 3:8] == 0.0)
    assert np.all(masked["x_teacher"][:, :3] == 1.0)
