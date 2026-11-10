import torch

from src.world_model.m3w_exchange_risk_heads import loss, success_log_probability


def test_risk_heads_only_use_labels_with_support():
    scores = torch.zeros(4, 7, requires_grad=True)
    status = torch.arange(4)
    loss(scores, torch.ones(4, dtype=torch.long), torch.zeros(4), status, True).backward()
    assert scores.grad[:, 4].ne(0.).sum() == 4
    assert scores.grad[:, 5].ne(0.).sum() == 3
    assert scores.grad[:, 6].ne(0.).sum() == 2
    assert scores.grad[0, 5] == scores.grad[1, 6] == 0.


def test_same_architecture_control_does_not_learn_risk_logits():
    scores = torch.zeros(2, 7, requires_grad=True)
    loss(scores, torch.ones(2, dtype=torch.long), torch.zeros(2), torch.zeros(2, dtype=torch.long), False).backward()
    assert scores.grad[:, 4:].eq(0.).all()


def test_conditional_probability_accounts_for_all_three_stages():
    scores = torch.zeros(2, 7)
    scores[1, 4:] = torch.tensor([-2., 8., 8.])
    p = success_log_probability(scores).exp()
    torch.testing.assert_close(p, torch.sigmoid(scores[:, 4:]).prod(1))
    assert p[0] > p[1]
