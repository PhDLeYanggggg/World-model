import numpy as np
import pytest
import torch
from src.world_model.m3w_fraction_square_head import fraction, loss_value, fit
from src.world_model.m3w_bounded_cost_head import build
from src.world_model.m3w_native_forecast import draw_batch


def test_loss_matches_two_output_weighted_fraction_square_not_native_or_three_part():
    logits = torch.tensor([[.2, -.5], [2., 1.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[.4, 0.], [0., .2], [0., 0.]], dtype=torch.float64)
    d, w = torch.tensor([2., 3., 0.]), torch.tensor([1., 4., 2.])
    p = np.logaddexp(0, logits.detach().numpy()); p /= 1+p.sum(1, keepdims=True)
    expected = np.mean(w.numpy()*d.numpy()*np.square(p-target.numpy()).mean(1))
    loss = loss_value(logits, target, d, w)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward(); assert torch.isfinite(logits.grad).all() and not logits.grad[-1].any()


def test_fraction_is_unchanged_cost_head_parameterization():
    torch.manual_seed(10)
    model = build(5, 8, 10); x = torch.randn(20, 5)
    torch.testing.assert_close(fraction(model.network(x)), model(x, torch.ones(20), 'bounded_native'), rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['negative', 'outside', 'missing', 'weight', 'distance'])
def test_invalid_loss_inputs_rejected(fault):
    logits = torch.zeros(3, 2); target = torch.zeros(3, 2); d = torch.ones(3); w = torch.ones(3)
    if fault == 'negative': target[0, 0] = -.1
    if fault == 'outside': target[0] = 1
    if fault == 'missing': target[0, 1] = float('nan')
    if fault == 'weight': w[0] = 0
    if fault == 'distance': d[0] = -1
    with pytest.raises(ValueError): loss_value(logits, target, d, w)


def test_real_training_resume_and_reference_draws_match(tmp_path):
    rng = np.random.default_rng(3); x = rng.normal(size=(30, 5)).astype(np.float32)
    sites = np.repeat(['a', 'b', 'c'], 10); known = np.ones(30, bool); known[[1, 15]] = False
    d = rng.uniform(.1, 3, 30); y = np.column_stack([.25*d, .05*d]); y[~known] = np.nan
    pr = dict(known=known, mean=np.zeros(5), std=np.ones(5), weights=np.ones(30), constant=np.zeros(5, bool),
              cost_scale=2., hard_cut=3., positive_easy_cut=1.)
    w = np.ones(30); w[~known] = 0; groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    settings = dict(steps=10, width=8, batch_size=6, learning_rate=.001, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    gen = torch.Generator().manual_seed(7+7919); draws = np.zeros(30, np.int64)
    for _ in range(settings['steps']): np.add.at(draws, draw_batch(groups, settings['batch_size'], gen), 1)
    kwargs = dict(seed=7, settings=settings, identity={'test': 1}, heartbeat=lambda **r: None)
    torch.set_num_threads(1)
    fit(x, y, d, sites, pr, w, draws, directory=tmp_path/'resume', stop_at=4, **kwargs)
    a, ar = fit(x, y, d, sites, pr, w, draws, directory=tmp_path/'resume', resume=True, **kwargs)
    b, br = fit(x, y, d, sites, pr, w, draws, directory=tmp_path/'full', **kwargs)
    for key in a.state_dict(): torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    assert ar['complete'] and ar['unknown_rows_sampled'] == 0 and ar['total_draws'] == 60
    assert [r['loss'] for r in ar['trace'] if r['step'] != 4] == [r['loss'] for r in br['trace'] if r['step'] != 4]
    with pytest.raises(ValueError, match='resume'):
        fit(x, y, d, sites, pr, w, draws, directory=tmp_path/'full', **kwargs)
