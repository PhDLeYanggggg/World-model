import numpy as np
import pytest

from src.evaluation.m3w_opportunity_diagnosis import causal_reasons, opportunity_ledger, risk_bands


def test_causal_reason_priority_and_zero_mass():
    u = np.array([1., 1., 0., 1., 1., 1.])
    m = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 0.], [1., .03], [1., .02]])
    motion = np.array([1, 0, 1, 1, 1, 1], bool)
    support = np.array([0, 1, 1, 1, 1, 1], bool)
    np.testing.assert_array_equal(causal_reasons(u, m, motion, support), np.arange(6))


def test_ledger_conserves_gain_harm_and_unknown_population():
    r = np.array([10., 10., 10., np.nan])
    c = np.array([5., 7., 12., np.nan])
    got = opportunity_ledger(r, c, np.array([4, 5, 5, 5]), np.array(['a']*4), expected_scenes=['a'])
    x = got['by_scene']['a']['contributions_percent']
    assert got['indexed_rows'] == 4 and got['unknown_rows'] == 1
    assert sum(got['by_scene']['a']['counts'].values()) == 4
    assert x['oracle_gain'] == pytest.approx(100*8/30)
    assert x['captured_gain'] == 10
    assert x['switched_harm'] == pytest.approx(100*2/30)
    assert x['net_gain'] == pytest.approx(100/30)
    assert x['oracle_regret'] == pytest.approx(100*7/30)
    assert x['missed_risk_veto'] == pytest.approx(100*5/30)


def test_empty_or_zero_reference_locality_is_not_dropped():
    for r, c in (([1.], [0.]), ([0., 1.], [1., 0.])):
        sites = np.array(['b'] if len(r) == 1 else ['a', 'b'])
        x = opportunity_ledger(np.array(r), np.array(c), np.full(len(r), 5), sites, expected_scenes=['a', 'b'])
        assert x['summary']['net_gain']['equal_locality'] is None
        assert x['summary']['net_gain']['ci95'] is None


def test_future_labels_change_attribution_not_causal_reasons():
    u = np.array([1., -1.]); m = np.array([[1., 0.], [1., 0.]])
    why = causal_reasons(u, m, np.ones(2, bool), np.ones(2, bool))
    a = opportunity_ledger(np.array([2., 2.]), np.array([0., 3.]), why, np.array(['a', 'a']), expected_scenes=['a'])
    b = opportunity_ledger(np.array([2., 2.]), np.array([4., 1.]), why, np.array(['a', 'a']), expected_scenes=['a'])
    np.testing.assert_array_equal(why, causal_reasons(u, m, np.ones(2, bool), np.ones(2, bool)))
    assert a['summary']['net_gain'] != b['summary']['net_gain']


def test_risk_bands_partition_unknown_and_boundary_rows():
    r = np.array([1., 1., 1., 1., 1., np.nan]); c = r.copy()
    m = np.array([[0, 0], [1, .02], [1, .1], [1, .5], [1, 1], [1, 2]])
    out = risk_bands(r, c, np.ones(6), m, np.zeros(6, dtype=int))
    assert all(v['indexed_rows'] == 1 for v in out.values())
    assert out['gt_1']['supported_rows'] == 0


def test_mismatched_label_support_rejected():
    with pytest.raises(ValueError):
        opportunity_ledger(np.array([np.nan]), np.array([0.]), np.array([5]), np.array(['a']), expected_scenes=['a'])
