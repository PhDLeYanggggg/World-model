import numpy as np

from src import stage41_joint_residual_domain_policy as rdp


def test_apply_sliced_policy_only_switches_matching_slice(monkeypatch):
    data = {
        "x": np.zeros((4, 2)),
        "domain": np.asarray(["A", "A", "B", "A"]),
        "horizon": np.asarray([50, 25, 50, 50]),
    }
    pred = {"score": np.arange(4)}

    def fake_switch(local_pred, params):
        return np.ones(len(local_pred["score"]), dtype=bool)

    monkeypatch.setattr(rdp.jrr, "_policy_switch", fake_switch)
    policy = {"slices": {"A|50": {"dummy": 1}}}
    switch = rdp._apply_sliced_policy(pred, data, policy)
    assert switch.tolist() == [True, False, False, True]


def test_slice_arrays_preserves_masked_rows():
    pred = {"a": np.asarray([1, 2, 3]), "b": np.asarray([4, 5, 6])}
    out = rdp._slice_arrays(pred, np.asarray([True, False, True]))
    assert out["a"].tolist() == [1, 3]
    assert out["b"].tolist() == [4, 6]
