import inspect
import numpy as np
import pytest
import torch
from src.training.m3w_prefix_forecast import objective, fit
from src.world_model.m3w_native_forecast import fit_trial, masked_objective


def test_fixed_prefix_loss_matches_direct_available_reference_and_fallback():
    p = torch.arange(72, dtype=torch.float32).reshape(3, 12, 2)/10
    m = torch.zeros(3, 12, dtype=torch.bool); m[0, [0, 2, 7]] = True; m[1, 9:] = True
    y = torch.full_like(p, float('nan')); y[m] = 0
    factors = torch.tensor([.5, 2., 1.])
    got, full = objective(p, y, m, factors, 'prefix_balanced')
    expected = []
    for i in range(3):
        if not m[i].any():
            expected.append(0.); continue
        d = torch.linalg.vector_norm(p[i], dim=-1)
        total = d[m[i]].mean()
        prefix = [d[:k][m[i, :k]].mean() for k in (1, 3, 6) if m[i, :k].any()]
        early = torch.stack(prefix).mean() if prefix else total
        expected.append((.5*total+.5*early)*factors[i])
    torch.testing.assert_close(got, torch.stack([torch.as_tensor(x) for x in expected]).mean())
    torch.testing.assert_close(full[1], torch.linalg.vector_norm(p[1, 9:], dim=-1).mean())
    assert full[2] == 0


def test_control_exact_and_unknown_gradient_zero():
    p = torch.ones(3, 12, 2, requires_grad=True)
    y = torch.zeros_like(p); m = torch.ones(3, 12, dtype=torch.bool)
    m[2] = False; y[2] = float('nan'); f = torch.ones(3)
    for arm in ('available12', 'prefix_balanced'):
        loss, full = objective(p, y, m, f, arm)
        if arm == 'available12':
            old, old_full = masked_objective(p, y, m, f)
            assert torch.equal(loss, old) and torch.equal(full, old_full)
        g = torch.autograd.grad(loss, p)[0]
        assert torch.isfinite(g).all() and not g[2].any()
    with pytest.raises(ValueError):
        objective(p, y, m, f, 'tuned_after_readout')


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.randn(12, 2)*.01)
    def forward(self, inputs):
        return inputs['baseline']+self.w


def population():
    g = np.zeros((12, 476), np.float32); g[:, 16:24] = np.arange(-7, 1)/12
    g[:, 332:356] = np.tile(np.c_[np.arange(1, 13), np.zeros(12)].ravel(), (12, 1))
    y = g[:, 332:356].reshape(-1, 12, 2)*1.1
    m = np.ones((12, 12), bool); m[1] = False; y[1] = np.nan
    d = dict(geometry=g, target=y, valid=m, sites=np.repeat(['a', 'b', 'c'], 4))
    f = dict(train_ids=np.arange(8), held_ids=np.arange(8, 12),
        groups=[np.arange(4), np.arange(4, 8)], factors=np.r_[np.ones(8), np.zeros(4)])
    return d, f


def equal_state(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_state(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_state(x, y)
    else: assert a == b


def test_new_control_engine_exact_to_original_and_prefix_resume(tmp_path):
    d, f = population()
    settings = dict(steps=4, batch_size=6, learning_rate=.001, weight_decay=.0001,
        minimum_lr_ratio=.01, checkpoint_every=2, heartbeat_every=1, gradient_clip=5.)
    def run(home, arm, legacy=False, resume=False, stop=None):
        torch.manual_seed(17); model = Tiny()
        kw = dict(seed=17, settings=settings, identity=dict(fixed=arm), directory=home,
            resume=resume, stop_at=stop, heartbeat=lambda **k: None)
        if legacy: fit_trial(model, d, f, **kw)
        else: fit(model, d, f, arm=arm, **kw)
        return torch.load(home/'checkpoint.pt', weights_only=False)
    a = run(tmp_path/'old', 'available12', True); b = run(tmp_path/'new', 'available12')
    for k in a:
        if k != 'seconds': equal_state(a[k], b[k])
    a = run(tmp_path/'full', 'prefix_balanced')
    run(tmp_path/'resume', 'prefix_balanced', stop=2)
    b = run(tmp_path/'resume', 'prefix_balanced', resume=True)
    for k in a:
        if k != 'seconds': equal_state(a[k], b[k])
    assert not b['draws'][8:].any()


def test_model_input_packing_has_no_label_arguments():
    from src.world_model.m3w_native_forecast import pack_geometry
    assert list(inspect.signature(pack_geometry).parameters) == ['geometry']


def test_paired_readout_retains_unknown_and_empty_easy_reference():
    from scripts.run_m3w_moving_prefix_actor import statistics
    old = np.ones((3, 12, 2)); new = old*.5; floor = np.zeros_like(old)
    y = np.ones_like(old)*2; m = np.ones((3, 12), bool)
    m[1, 3:] = False; m[2] = False; y[~m] = np.nan
    rows = statistics(old, new, floor, y, m, np.ones(3)/3, .01, 1.)
    full = rows[-1]
    assert full['rows'] == 3 and full['known_rows'] == 2
    assert full['wholly_unknown_rows'] == 1 and full['partial_rows'] == 1
    assert full['paired_completion_lower_pp'] < full['paired_known_pp']
    assert full['treatment']['known_easy_positive_harm_percent'] is None
    assert full['treatment']['known_easy_signed_degradation_percent'] is None
    assert full['paired_known_pp'] < 0
