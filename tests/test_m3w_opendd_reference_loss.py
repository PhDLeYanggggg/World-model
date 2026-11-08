import numpy as np
import torch

from src.world_model import m3w_opendd_reference_loss as m


def test_loss_requires_removing_the_only_existing_reference():
    np.testing.assert_array_equal(m.reference_targets(False, True, [False, True]),
                                  [[True, True, True], [False, True, False]])
    for gain, old in ((True, True), (True, False), (False, False)):
        assert not m.reference_targets(gain, old, [False, True])[:, 0].any()


def test_hazard_factorization_and_exact_control_count():
    np.testing.assert_allclose(m.scores(np.zeros((3, 3))), [[.5, .25]]*3)
    np.testing.assert_array_equal(m.matched_easy([.7, .8, .1], [True, False, False]),
                                  [False, True, False])


def test_loss_finite_without_any_old_unique_reference():
    net = m.network(4)
    value = m.loss(net, torch.zeros((3, 4)), torch.zeros((3, 3)))
    assert torch.isfinite(value)
    value.backward()
