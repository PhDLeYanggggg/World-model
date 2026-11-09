import torch

from src.world_model import m3w_factored_counterfactual_support as model


def test_factorization_is_bounded_and_empty_conditional_loss_is_finite():
    z = torch.tensor([[0., 0., 0.], [0., 0., 2.]])
    local, whole = model.risks(z)
    torch.testing.assert_close(local, torch.tensor([.5, .5]))
    assert (whole <= local).all()
    assert whole[0] == .25
    net = model.Value(5)
    loss = model.objective(net, torch.zeros(2,5), torch.zeros(2,3))
    assert torch.isfinite(loss)
    loss.backward()
