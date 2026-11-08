import subprocess
import sys

import numpy as np
import torch

from src.world_model import m3w_opendd_gain_wait as model


def test_first_legal_offer_consumes_no_suffix():
    def stream():
        yield False, np.array([2]), np.array([4.]), np.array([7.])
        yield True, np.array([3, 4, 5]), np.array([.9, 1., 3.]), np.array([9., 2., 1.])
        raise AssertionError("Read an offer after committing")

    assert model.first_offer(stream(), 1.) == (2, 4)


def test_expiry_never_consumes_an_extra_query():
    def stream():
        for _ in range(model.DEADLINE):
            yield True, np.array([2]), np.array([.5]), np.array([7.])
        raise AssertionError("Read an offer beyond the deadline")

    assert model.first_offer(stream(), 1.) is None


def test_guard_and_loss():
    scores = np.array([[1., -1., -1.], [1., 1., -1.], [1., -1., 1.], [0., -1., -1.]])
    np.testing.assert_array_equal(model.choose_wait(scores, True), [True, False, False, False])
    np.testing.assert_array_equal(model.choose_wait(scores, False), [True, True, True, False])
    net = model.network(4)
    loss = model.loss(net, torch.ones((4, 4)), torch.zeros((4, 3)))
    assert torch.isfinite(loss)
    loss.backward()


def test_release_features_do_not_reveal_future_support_or_queries():
    # The runner's historical imports exceed pytest's assertion-rewrite stack.
    subprocess.run([sys.executable, "-c", """
import numpy as np
from scripts.run_m3w_opendd_gain_wait import release_context
seq = [np.array([0]), np.array([1]), np.array([2])]
q, sizes = np.arange(3), np.ones(3)
feat = [np.ones((3, 4)), np.zeros((3, 4))]
before, width, known = release_context(seq, 1, 1, 2, q, feat, sizes)
feat[1][2] = 10.
sizes[2] = 20.
after, _, _ = release_context(seq+[np.array([2])], 1, 1, None, q, feat, sizes)
np.testing.assert_array_equal(before, after)
assert not known
assert width == 5
"""], check=True)
