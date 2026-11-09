import torch

from src.world_model import m3w_av2_replay_hurdle as model


def test_conditional_loss_is_masked_when_the_anchor_is_ineligible():
    logits = torch.zeros(2, 3, requires_grad=True)
    y = torch.tensor([[0., 0.], [1., 1.]])
    model.loss(logits, y).backward()
    assert logits.grad[0, 2] == 0.
    assert logits.grad[1, 2] < 0.
    assert logits.grad[0, 1] > 0.


def test_reference_score_is_candidate_invariant_and_factored_bounded():
    torch.manual_seed(11)
    net = model.ReplayHurdle(3)
    a = torch.zeros(2, 3)
    p = torch.stack((torch.zeros(3), torch.ones(3)))
    replay = torch.tensor([.2, .4])
    logits = net(a, p, replay)
    assert logits[0, 0] == logits[1, 0]
    values = model.scores(logits, replay)
    torch.testing.assert_close(values[:, 3], values[:, 4]*values[:, 5])
    assert (values[:, 3] <= values[:, 4]).all()
