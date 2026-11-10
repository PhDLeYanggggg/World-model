import pytest
import torch

from src.world_model import m3w_vehicle_stgcnn as model


def test_laplacian_respects_missing_neighbors_and_published_kernel():
    motion = torch.tensor([[[[0., 0.]], [[2., 0.]], [[4., 0.]]]])
    a = model.graph(motion, torch.tensor([[True, True, False]]))
    expected = torch.tensor([[[[1/3, -1/3, 0], [-1/3, 1/3, 0], [0, 0, 0]]]])
    torch.testing.assert_close(a, expected)


@pytest.mark.skipif(not all((model.SOURCE/k).is_file() for k in model.DIGESTS), reason='Pinned author source not acquired')
def test_batched_author_model_matches_the_original_graph_execution():
    torch.manual_seed(17)
    upstream = model.author().social_stgcnn(**model.ARCH).eval()
    new = model.Forecast('social_peers').eval().net
    new.load_state_dict(upstream.state_dict())
    x = torch.randn(3, 2, 8, 4)
    a = model.graph(x.permute(0, 3, 2, 1), torch.ones(3, 4, dtype=torch.bool))
    with torch.inference_mode():
        batch, _ = new(x, a)
        one = torch.cat([upstream(x[i:i+1], a[i])[0] for i in range(len(x))])
    torch.testing.assert_close(batch, one, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not all((model.SOURCE/k).is_file() for k in model.DIGESTS), reason='Pinned author source not acquired')
def test_padded_history_never_affects_forecast():
    net = model.Forecast('social_peers').eval()
    e, p = torch.randn(2, 8, 2), torch.randn(2, 3, 8, 2)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    with torch.inference_mode():
        a = net(e, p, mask)
        b = net(e, torch.where(mask[..., None, None], p, p+1e6), mask)
    assert a.shape == (2, 12, 2)
    torch.testing.assert_close(a, b)
