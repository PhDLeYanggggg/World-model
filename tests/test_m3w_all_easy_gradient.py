import numpy as np
import torch
from src.evaluation import m3w_all_easy_gradient as api
from tests.test_m3w_admission_priority import prepared


def test_tasks_match_existing_loss_and_unknowns(tmp_path):
    torch.set_num_threads(1); d, warm, _ = prepared(tmp_path)
    pred = torch.randn(4, 7, 2, requires_grad=True); seg = torch.tensor([0, 0, 1, 1]); ix = np.arange(4)
    old = api.kin.parent.objective(pred, d, ix, seg, 2, 'mean')['total']
    y = api.kin.parent.signed(d['truth_by_arm']['bound_aware'])/torch.from_numpy(api.kin.parent.normalizer(d['preprocess']))
    weights = api.query_weights(d['causal'], np.array([1, 1]))
    pair = api.task_losses(pred, y, torch.from_numpy(d['causal']['allowed']), torch.from_numpy(weights))
    torch.testing.assert_close(pair.mean(), old.double(), rtol=1e-6, atol=1e-6)
    g0 = torch.autograd.grad(old, pred, retain_graph=True)[0]
    g1 = torch.autograd.grad(pair.mean(), pred)[0]
    torch.testing.assert_close(g0, g1)


def test_full_gradients_and_virtual_preserve_parent(tmp_path):
    torch.set_num_threads(1); d, warm, _ = prepared(tmp_path)
    model = api.kin.KinematicRiskHead(4, 17); model.load_state_dict(warm['model'])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses, grads = api.full(model, d, warm['sampler']['counts'], True)
    assert np.isfinite(losses['task_MSE']).all()
    for variant in api.VARIANTS:
        result, detail = api.virtual(model, warm, grads, variant)
        assert np.isfinite(detail['gradient_norm_before_clip'])
        assert any(not torch.equal(v, before[k]) for k, v in result.state_dict().items())
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_protected_combination_removes_negative_all_component():
    model = api.kin.KinematicRiskHead(4, 17)
    a = [torch.full_like(p, -3, dtype=torch.float64) for p in model.parameters()]
    e = [torch.ones_like(p, dtype=torch.float64) for p in model.parameters()]
    gs, relation = api.combine(model, (a, e), 'easy_protected')
    assert relation['joint_first_order_easy_harm']
    for (name, _), g in zip(model.named_parameters(), gs):
        torch.testing.assert_close(g, torch.full_like(g, -1 if name.startswith('excess.') else .5))
