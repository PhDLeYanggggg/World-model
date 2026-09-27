import numpy as np
import pytest
import torch
from src.world_model import m3w_oof_magnitude as m
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_membership_auxiliary import AuxiliaryCostHead


def fixture():
    x = np.arange(80, dtype=float).reshape(20, 4)
    cv = np.arange(20, dtype=float)+1
    raw = np.column_stack([cv, cv/2])
    return x, raw, cv, np.array(['a']*10+['b']*10)


def test_single_site_reference_has_no_hidden_site_statistics():
    x, y, cv, sites = fixture(); take = sites == 'a'
    a = m.reference_preprocess(x[take], y[take], cv[take], sites[take], ['b', 'c'])
    x[~take] = 1e12; cv[~take] = 1e13
    b = m.reference_preprocess(x[take], y[take], cv[take], sites[take], ['b', 'c'])
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert a['training_sites'] == ['a']
    np.testing.assert_allclose(a['mean'], x[take].mean(0))
    with pytest.raises(ValueError): m.reference_preprocess(x, y, cv, sites, ['b'])


def test_multisite_preprocessing_preserves_original():
    x, y, cv, sites = fixture()
    a = m.reference_preprocess(x, y, cv, sites, ['c'])
    b = preprocess(x, y, cv, sites, 'c')
    for key in a: np.testing.assert_array_equal(a[key], b[key])


def test_reference_unknown_rows_excluded():
    x, y, cv, sites = fixture(); sites[:] = 'a'
    y[-1] = np.nan; cv[-1] = np.nan; x[-1] = 1e15
    p = m.reference_preprocess(x, y, cv, sites, ['b'])
    assert p['weights'][-1] == 0
    np.testing.assert_allclose(p['mean'], x[:-1].mean(0))


def test_nested_reference_rejects_inner_held_leakage():
    m.check_lineage(['a'], ['b'], 'd', ['e'])
    for prediction, fit in [(['a'], ['a', 'b']), (['a'], ['b', 'd']), (['a'], ['e'])]:
        with pytest.raises(ValueError): m.check_lineage(prediction, fit, 'd', ['e'])


def test_origin_slopes_and_nested_envelope():
    p = np.tile([4., 2., 2., 1.], (8, 1)); y = p.copy(); y[:, 1] = 4; y[:, 3] = 3
    env = np.full(8, 5.); sites = np.array(['a']*4+['b']*4)
    head = m.fit_magnitude(p, y, env, sites, 'c')
    assert head['slopes'] == [2., 3.]
    got = m.predict_magnitude(head, p, env)
    np.testing.assert_array_equal(got, y)
    env[0] = 0; got = m.predict_magnitude(head, p, env)
    assert got[0, 1] == got[0, 3] == 0
    np.testing.assert_array_equal(got[:, [0, 2]], p[:, [0, 2]])
    assert np.all(got[:, 3] <= got[:, 1]) and np.all(got[:, 1] <= env)


def test_magnitude_rejects_outer_and_ignores_unknown_labels():
    p = np.ones((8, 4)); y = p.copy(); y[-1] = np.nan
    sites = np.array(['a']*4+['b']*4); env = np.ones(8)
    head = m.fit_magnitude(p, y, env, sites, 'c')
    assert head['slopes'] == [1., 1.] and head['known_rows'] == 7
    with pytest.raises(ValueError): m.fit_magnitude(p, y, env, sites, 'a')
    y[-1, 0] = 1
    with pytest.raises(ValueError): m.fit_magnitude(p, y, env, sites, 'c')


def test_compression_is_lossless_and_restorable(tmp_path):
    model = AuxiliaryCostHead(4, 8)
    state = dict(preprocess=dict(mean=np.zeros(4)), settings=dict(width=8), model=model.state_dict())
    torch.save(state, tmp_path/'checkpoint.pt')
    before = (tmp_path/'checkpoint.pt').read_bytes()
    path = m.compress_checkpoint(tmp_path)
    import gzip
    assert gzip.decompress(path.read_bytes()) == before
    restored, got = m.restore(tmp_path)
    for key in model.state_dict(): assert torch.equal(restored.state_dict()[key], model.state_dict()[key])
    assert not (tmp_path/'checkpoint.pt').exists()
