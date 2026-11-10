import numpy as np
import pytest
import torch

from src.world_model import m3w_recipient_eligibility as model


def test_rank_uses_raw_movement_and_eligibility_without_changing_count():
    assert model.rank(np.array([.9, .6]), np.array([.8, .2]), np.array([1., 2.])) == (1, 0, 0)
    assert model.rank(np.array([.9, .6]), np.array([.4, .2]), np.array([1., 2.])) == (1, 0, -1)
    assert model.rank(np.array([-.9, -.6]), np.array([.8, .7]), np.array([1., 2.])) == (0, 0, -1)


def test_nonfinite_scores_cannot_trigger_swaps():
    with pytest.raises(ValueError, match='finite'):
        model.rank(np.array([float('nan')]), np.array([.8]), np.array([1.]))


def test_eligibility_and_gain_losses_have_separate_targets():
    pred = torch.zeros((2, 2), requires_grad=True)
    y = torch.tensor([[.5, 0.], [-.5, 1.]])
    total, gain, eligible = model.loss(pred, y)
    assert torch.equal(total, gain+eligible)
    total.backward()
    torch.testing.assert_close(torch.sign(pred.grad), torch.tensor([[-1., 1.], [1., -1.]]))


def test_propose_matches_the_archived_mask_without_target_fields():
    from scripts import run_m3w_opendd_recipient_eligibility as run

    h = np.arange(-7., 1.)[None, :, None]*np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]]])
    cv = np.zeros((6, 12, 2))
    anchor = np.zeros((2, 2, 12, 2))
    anchor[..., 0] = .5
    f = dict(counts=[3, 3], history=np.tile(h, (2, 1, 1)), offsets=np.zeros((6, 2)), x=np.ones((6, 4)))
    p = dict(cv=cv, bounds=[[0, 3], [3, 6]], additive_rows=np.array([[0, 1], [3, 4]]), additive_paths=anchor)

    class Head(torch.nn.Module):
        def forward(self, x):
            assert len(x) == 2
            return torch.tensor([[.3, -2.], [.1, 2.]])

    a = run.propose(p, f, Head(), np.array([-1, 0]))
    assert a['gain'].tolist() == [-1, 0]
    assert a['eligibility'].tolist() == a['guarded'].tolist() == [-1, 1]
