import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.world_model import m3w_temporal_target_audit as api


class Tree:
    def apply(self, x):
        return (x[:, 0] > 0).astype(int)


def fixture():
    state = dict(preprocess=dict(mean=np.zeros(1), std=np.ones(1), clip=8.,
                 scale=1., support_limit=100.), model=SimpleNamespace(estimators_=[Tree()]))
    x = np.array([[-2.], [-1.], [1.], [2.]])
    v = np.stack([np.tile([i, 2*i], (12, 1)) for i in (1., 3., 5., 7.)])
    v[0, 6:] = np.nan
    return state, x, np.ones(4), v, np.repeat('s', 4), np.arange(4), np.zeros(4, int)


def test_cancellation_preserves_original_harm_not_average_step_harm():
    ref = np.zeros((3, 12, 2)); ref[..., 0] = 2
    pred = ref.copy(); pred[:, :6, 0] = 4; pred[:, 6:, 0] = 0
    target = np.zeros_like(ref); valid = np.ones((3, 12), bool)
    valid[1, 6:] = False; valid[2] = False; target[~valid] = np.nan
    series, d = api.targets(ref, pred, target, valid)
    np.testing.assert_array_equal(d['harm'][:2], [0., 2.])
    np.testing.assert_array_equal(d['gross_harm'][:2], [1., 2.])
    np.testing.assert_array_equal(d['cancellation'][:2], [1., 0.])
    assert np.isnan(d['harm'][2]) and np.isnan(series[~valid]).all()
    assert d['opposite_sign'][0] and not d['opposite_sign'][1]


def test_temporal_leaf_means_and_train_only_missingness():
    args = fixture(); model = api.fit(*args)
    pred, support = api.predict(args[0], model, args[1], args[2])
    np.testing.assert_array_equal(pred['temporal_leaf'][0, :6, 0], 2.)
    np.testing.assert_array_equal(pred['temporal_leaf'][0, 6:, 0], 3.)
    assert support.all()
    assert list(inspect.signature(api.predict).parameters) == ['state', 'fitted', 'x', 'envelope']
    assert api.fingerprint(model) == api.fingerprint(api.fit(*args))


def test_unobserved_training_step_is_not_safely_zero_filled():
    args = list(fixture()); args[3][:, 11] = np.nan
    model = api.fit(*args); pred, support = api.predict(args[0], model, args[1], args[2])
    assert not support[:, 11].any()
    for p in pred.values(): assert np.isnan(p[:, 11]).all()


def test_probe_score_matches_scalar_and_retains_unknown_counts():
    args = fixture(); truth = args[3].copy(); truth[3] = np.nan
    pred, support = api.predict(args[0], api.fit(*args), args[1], args[2])
    s = api.score(pred, truth, args[4], args[5], args[6])
    assert s['unknown_rows'] == 1
    for name, p in pred.items():
        rows = []
        for i in range(3):
            ix = np.isfinite(truth[i, :, 0])
            rows.append(((p[i, ix]-truth[i, ix])**2).mean(0))
        np.testing.assert_allclose(s[name], np.mean(rows, 0))


def test_masked_auxiliary_loss_balances_rows_then_queries():
    p = torch.zeros((3, 12, 2), requires_grad=True)
    y = torch.ones_like(p); mask = torch.ones((3, 12), dtype=torch.bool)
    y[1] = 3.; y[2] = 2.; mask[1, 1:] = False
    y[~mask] = float('nan')
    loss = api.auxiliary_loss(p, y, mask, torch.tensor([0, 0, 1]), 2)
    assert loss.item() == 4.5
    loss.backward(); assert torch.isfinite(p.grad).all()
    assert not p.grad[~mask].any()


def test_auxiliary_detaches_labels_and_rejects_empty_query():
    p = torch.ones((1, 12, 2), requires_grad=True)
    y = torch.zeros_like(p, requires_grad=True)
    m = torch.ones((1, 12), dtype=torch.bool)
    api.auxiliary_loss(p, y, m, torch.tensor([0]), 1).backward()
    assert y.grad is None
    with pytest.raises(ValueError):
        api.auxiliary_loss(p, y, m, torch.tensor([0]), 2)
    with pytest.raises(ValueError):
        api.auxiliary_loss(p, y, torch.zeros_like(m), torch.tensor([0]), 1)


@pytest.mark.parametrize('case', ['shape', 'nan_prediction', 'bad_mask', 'bad_target'])
def test_temporal_target_invalid_inputs_fail(case):
    r = np.zeros((2, 12, 2)); p = r.copy(); y = r.copy(); m = np.ones((2, 12), bool)
    if case == 'shape': p = p[:, :8]
    if case == 'nan_prediction': p[0, 0, 0] = np.nan
    if case == 'bad_mask': m = m.astype(int)
    if case == 'bad_target': y[0, 0, 0] = np.nan
    with pytest.raises(ValueError): api.targets(r, p, y, m)
