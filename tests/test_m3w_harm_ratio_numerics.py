import pytest

from src.world_model import m3w_isolated_harm as old
from src.world_model import m3w_isolated_harm_fp64 as new

torch = new.torch


class TinyBase(torch.nn.Module):
    def __init__(self, envelope):
        super().__init__()
        self.primary = torch.nn.Module()
        self.primary.encoder = torch.nn.Identity()
        self.moments = torch.nn.Parameter(torch.tensor([[0., envelope/2, 1., 1., envelope/4]]))

    def forward(self, z, envelope):
        return self.moments, None


@pytest.mark.parametrize('envelope', [0., 1e-38, 1e-30, 1e-20, 1., 1e20, 1e30])
@pytest.mark.parametrize('bias', [-2., 0., 2.])
def test_bounded_ratio_extremes_backward_remains_finite(envelope, bias):
    model = new.HarmBranch(TinyBase(envelope), 2, False)
    with torch.no_grad():
        model.risk_output.bias.fill_(bias)
    out = model(torch.zeros(1, 2), torch.tensor([envelope]))
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert out.dtype == torch.float32
    assert out[0, 0]+out[0, 1] <= envelope
    assert out[0, 4] <= out[0, 1]


def test_float32_precursor_reproduces_nan_backward():
    model = old.HarmBranch(TinyBase(0.), 2, False)
    with torch.no_grad():
        model.risk_output.bias.fill_(2.)
    out = model(torch.zeros(1, 2), torch.zeros(1))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
