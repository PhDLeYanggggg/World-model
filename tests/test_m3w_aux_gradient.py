import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_aux_gradient as m
from src.world_model.m3w_membership_auxiliary import AuxiliaryCostHead


def fixture():
    torch.manual_seed(31)
    model = AuxiliaryCostHead(5, 7)
    x, d = torch.randn(25, 5), torch.linspace(0, 2, 25)
    y, e = torch.rand(25, 4), (torch.arange(25) % 3 == 0).float()
    e[0] = float('nan')
    scales = torch.ones(4)
    settings = dict(learning_rate=.0003, gradient_clip=5.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    cost = ((model(x, d)[0] - y)**2).mean()
    cost.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return model, x, d, y, e, scales, settings, optimizer


def test_projection_removes_negative_shared_component_only():
    main = [torch.tensor([2., 0.]), torch.tensor([5.])]
    aux = [torch.tensor([-3., 4.]), torch.tensor([-9.])]
    before = copy.deepcopy(aux)
    out = m.projected(main, aux, [0])
    assert float(out[0] @ main[0]) >= 0
    torch.testing.assert_close(out[0], torch.tensor([0., 4.]))
    for a, b in zip(aux, before): assert torch.equal(a, b)
    assert torch.equal(out[1], aux[1])


def test_zero_gradient_is_not_estimable_not_agreement():
    assert m.geometry(torch.zeros(3), torch.ones(3))['cosine'] is None
    a = [torch.zeros(2)]; b = [torch.ones(2)]
    assert torch.equal(m.projected(a, b, [0])[0], b[0])


def test_rows_disjoint_known_deterministic():
    known = np.arange(800) % 5 != 0
    sites = np.array(['a']*400 + ['b']*400)
    a, p = m.sample_rows(known, sites, 23, 0, 64, 32)
    b, q = m.sample_rows(known, sites, 23, 0, 64, 32)
    np.testing.assert_array_equal(a, b)
    assert known[a].all()
    for site in p:
        np.testing.assert_array_equal(p[site], q[site])
        assert known[p[site]].all() and not np.intersect1d(a, p[site]).size
    with pytest.raises(ValueError): m.sample_rows(known, sites, 1, 0, 64, 500)


@pytest.mark.parametrize('variant', m.VARIANTS)
def test_virtual_step_preserves_checkpoint_and_moments(variant):
    model, x, d, y, e, s, cfg, opt = fixture()
    original = copy.deepcopy(model.state_dict()); state = copy.deepcopy(opt.state_dict())
    gs, _ = m.gradients(model, x, d, y, e, s)
    shared = [i for i, (n, _) in enumerate(model.named_parameters()) if n.startswith('network.0.')]
    a, _ = m.virtual_step(model, opt.state_dict(), cfg, gs[0], gs[2], shared, variant)
    b, _ = m.virtual_step(model, opt.state_dict(), cfg, gs[0], gs[2], shared, variant)
    for k in original:
        assert torch.equal(model.state_dict()[k], original[k])
        assert torch.equal(a.state_dict()[k], b.state_dict()[k])
    for k in state['state']:
        for name, value in state['state'][k].items():
            assert torch.equal(value, opt.state_dict()['state'][k][name])


def test_true_aux_matches_real_adam_objective_update():
    model, x, d, y, e, s, cfg, opt = fixture()
    expected = copy.deepcopy(model)
    real = torch.optim.AdamW(expected.parameters(), lr=.0003, weight_decay=.0001)
    real.load_state_dict(copy.deepcopy(opt.state_dict()))
    p, l = expected(x, d); mask = torch.isfinite(e)
    loss = ((p-y)**2).mean()+torch.nn.functional.binary_cross_entropy_with_logits(l[mask], e[mask])
    loss.backward(); torch.nn.utils.clip_grad_norm_(expected.parameters(), 5.); real.step()
    g, _ = m.gradients(model, x, d, y, e, s)
    actual, _ = m.virtual_step(model, opt.state_dict(), cfg, g[0], g[2], [0, 1], 'true_aux')
    for k in expected.state_dict():
        torch.testing.assert_close(actual.state_dict()[k], expected.state_dict()[k], rtol=1e-6, atol=1e-7)


def test_missing_auxiliary_and_positive_support():
    model, x, d, y, e, s, _, _ = fixture()
    g, info = m.gradients(model, x, d*0, y, e*float('nan'), s)
    assert info['auxiliary_BCE'] is None and info['easy_harm_positive'] is None
    assert all(torch.count_nonzero(a) == 0 for a in g[2])
    assert m.probe_losses(model, x, d*0, y, s)['easy_harm_positive'] is None
    y[0, 0] = float('nan')
    with pytest.raises(ValueError): m.gradients(model, x, d, y, e, s)


def test_initial_zero_head_has_no_shared_gradient():
    model, x, d, y, e, s, _, _ = fixture()
    with torch.no_grad():
        model.network[-1].weight.zero_(); model.membership.weight.zero_()
    g, _ = m.gradients(model, x, d, y, e, s)
    assert m.geometry(m.flat(g[0], [0, 1]), m.flat(g[2], [0, 1]))['cosine'] is None
    assert float(m.flat(g[0], list(range(len(g[0])))).norm()) > 0


def test_interval_unit_missing_support_and_direction():
    from scripts.report_m3w_european_aux_gradient import interval
    cfg = dict(bootstrap_seed=123, bootstrap_resamples=3000)
    a = interval([[.9, 1.]]*4, cfg)
    assert a['point'] == pytest.approx(10)
    assert a['CI'] == pytest.approx([10, 10])
    assert interval([[1., .9]]*4, cfg)['CI'][1] < 0
    assert interval([[.9, 1.]]*3, cfg)['status'] == 'not_estimable'
