import torch

from scripts import run_m3w_av2_easy_gate as run
from src.world_model import m3w_av2_easy_gate as model


def test_no_edited_easy_support_is_not_a_pass():
    assert run.easy_events(dict(completion_easy_risk_upper=None, known_easy_net_degradation=0.)) == (False, False)
    assert run.easy_events(dict(completion_easy_risk_upper=.01, known_easy_net_degradation=.021)) == (True, False)
    assert run.easy_events(dict(completion_easy_risk_upper=.02, known_easy_net_degradation=.02)) == (True, True)


def test_unsupported_rows_do_not_supervise_conditional_heads():
    z = torch.zeros(2, 5, requires_grad=True)
    y = torch.tensor([[0., 0., 3., 4.], [1., 1., 3., 4.]])
    model.loss(z, y).backward()
    torch.testing.assert_close(z.grad[0, 2:], torch.zeros(3))
    assert torch.count_nonzero(z.grad[1, 2:]) == 3
    assert (model.probabilities(z)[:, 2] <= model.probabilities(z)[:, 0]).all()


def test_edit_recipient_permutation_does_not_change_gate():
    net = model.EasyGate(7)
    edits, context = torch.randn(3, 2, 7), torch.randn(3, 42)
    torch.testing.assert_close(net(edits, context), net(edits.flip(1), context))
