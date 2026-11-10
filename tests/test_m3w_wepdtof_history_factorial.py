import numpy as np

from scripts import audit_m3w_wepdtof_history_factorial as study


def test_refit_uses_the_same_velocity_and_recovers_a_line():
    h = np.arange(8.)[None, :, None]*np.array([2., -1.])
    _, v = study.old.method.analytic(h)
    np.testing.assert_allclose(study.refit(h), v)
    h[:, -1, 0] += 4.
    _, v = study.old.method.analytic(h)
    d = study.refit(h)-v
    np.testing.assert_allclose(d, np.broadcast_to(d[:, :1], d.shape))
    assert d[0, 0, 0] < 0.


def test_crossed_controls_change_only_the_named_component():
    h = np.arange(8.)[None, :, None]*np.array([2., -1.])
    a = h+3.
    x = study.crossed(h, a)
    for v in x.values():
        np.testing.assert_allclose(v['detector'], v['annotation_velocity'])
        np.testing.assert_allclose(v['annotation_anchor'], v['annotation_both'])
        np.testing.assert_allclose(v['annotation_both']-v['detector'], 3.)


def test_attribution_closes_exactly_and_bootstraps_sources_not_clips():
    z = study.factor(dict(detector=10., annotation_anchor=9., annotation_velocity=7., annotation_both=4.))
    assert z['anchor']+z['velocity'] == z['total'] == 6.
    x = dict(a=dict(source='s1', val=4.), b=dict(source='s1', val=6.), c=dict(source='s2', val=1.))
    s = study.contrast(x, ['s1', 's2', 's3'], lambda r: r['val'])
    assert s['point'] == 3.
    assert s['estimable_sources'] == 2
    assert s['contributing_clips'] == {'s1': 2, 's2': 1, 's3': 0}
