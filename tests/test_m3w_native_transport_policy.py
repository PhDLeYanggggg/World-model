import numpy as np

from src.world_model.m3w_native_transport_policy import MATCHED, decide, gate


def test_gate_requires_positive_utility_and_reference_without_identity_exception():
    choice = np.array([0, 1, -1])
    gain = np.ones((3, 2))
    risk = np.zeros_like(gain)
    np.testing.assert_array_equal(gate(choice, gain, risk, risk, np.ones_like(risk)), choice)
    for g, a, e, ref in ((gain, risk, risk, risk), (-gain, risk, risk, gain),
                         (gain, gain, risk, gain), (gain, risk, gain, gain)):
        assert (gate(choice, g, a, e, ref) == -1).all()


def test_four_arms_use_identical_counts_and_gate_uses_same_joint_actions():
    n, k = 6, 2
    h = np.zeros((n, 8, 2))
    h[:, :, 0] = np.arange(n)[:, None] + np.arange(8)[None, :]
    floor = np.broadcast_to(np.arange(1, 13)[None, :, None], (n, 12, 2))
    bank = floor[:, None] + np.arange(1, k+1)[None, :, None, None]
    from src.world_model.m3w_record_joint_actions import descriptors
    x, _ = descriptors(h, bank, floor, np.full(n, n))
    c = np.zeros((k, x.shape[2], 6))
    c[:, 0, [0, 3, 4]] = [.1, 1., .2]
    out = decide(h, bank, floor, floor, np.full(n, n), c)
    assert {(out[a] >= 0).sum() for a in MATCHED} == {2}
    np.testing.assert_array_equal(out['gated_joint'], out['joint'])
