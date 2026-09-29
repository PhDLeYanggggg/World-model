import numpy as np
import pytest
from src.world_model.m3w_boundary_diagnostic import decompose, boundary_bins, diagnostic


def test_excess_identity_separates_harm_and_reference():
    y = np.array([[0., 3., 10., 10., 3.], [1., 0., 5., 0., 0.]])
    p = np.array([[0., .2, 20., 15., .1], [0., .1, 10., 1., 0.]])
    d = decompose(y, p, np.ones(2, bool), 1.)
    assert d['true_excess'] == pytest.approx(2.7)
    assert d['predicted_excess'] == pytest.approx(-.3)
    assert d['harm_underestimate'] == pytest.approx(2.7)
    assert d['reference_overestimate'] == pytest.approx(.3)
    assert d['realized_risk_ratio'] == pytest.approx(.2)


def test_unknown_and_zero_denominator_never_certify():
    y = np.full((1, 5), np.nan); p = np.ones((1, 5))
    d = decompose(y, p, np.ones(1, bool), 1.)
    assert d['unknown_selected'] == 1 and d['realized_risk_ratio'] is None
    assert not d['risk_certified']
    d = decompose(np.zeros((1, 5)), p, np.ones(1, bool), 1.)
    assert d['realized_risk_ratio'] is None


def test_tail_uses_supplied_training_cut_not_eval_quantile():
    y = np.array([[0., 1., 10., 0., 0.], [0., 10., 10., 0., 0.]])
    p = np.zeros((2, 5)); p[:, 2] = 10.
    d = decompose(y, p, np.ones(2, bool), 5.)
    assert d['tail_rows'] == 1 and d['tail_harm_share'] == pytest.approx(10/11)


def test_fixed_bins_partition_and_preserve_unknown():
    p = np.array([[0., 0., 10., 0., 0.], [0., .2, 10., 0., 0.], [0., 1., 10., 0., 0.]])
    y = p.copy(); y[1] = np.nan
    ix, bins = boundary_bins(y, p, np.ones(3, bool), 1., [-.1, 0., .1])
    assert ix.tolist() == [0, 2, 3]
    assert sum(b['unknown_rows'] for b in bins) == 1


def test_partially_unknown_target_rejected():
    y = np.zeros((1, 5)); y[0, 1] = np.nan
    with pytest.raises(ValueError):
        decompose(y, np.zeros((1, 5)), np.ones(1, bool), 0.)


def test_empty_selection_does_not_drop_roster():
    d = decompose(np.zeros((3, 5)), np.zeros((3, 5)), np.zeros(3, bool), 0.)
    assert d['rows'] == 3 and d['selected_rows'] == 0 and d['realized_risk_ratio'] is None


def packet():
    y = np.tile([0., .03, 1., 1., .03], (4, 1)); y[-1] = np.nan
    p = np.tile([.1, .005, 1., .5, .001], (4, 1))
    a = dict(ids=np.arange(4), targets=y, envelope=np.ones(4), moving=np.ones(4, bool),
             support=np.ones(4, bool), recording=np.zeros(4, int), frame=np.array([1, 1, 2, 2]))
    for arm in ('affine', 'nonlinear'):
        a.update({arm+'_pred': p.copy(), arm+'_selected': np.ones(4, bool), arm+'_matched': np.ones(4, bool)})
    meta = dict(phase='internal_transfer',site='b',train_site='a',outer_held_sites=['c','d'],
                source_roles=dict(producer_sites=['p'],controller_sites=['q']),cost_scale=1.,tail_cuts=dict(all=.02,easy=.02))
    cfg = dict(arms=['affine','nonlinear'],boundary_half_width=.02,bin_edges=[-.1,-.02,0.,.02,.1])
    return a,meta,cfg


def test_whole_packet_independent_arithmetic_and_no_parameter_updates():
    from scripts.run_m3w_boundary_diagnostic import verify_result
    a,meta,cfg = packet(); d = diagnostic(a,meta,cfg)
    assert verify_result(d) > 100
    assert d['parameter_updates'] == 0 and d['unknown_rows'] == 1
    assert d['arms']['affine']['scopes']['matched']['all']['realized_risk_ratio'] == pytest.approx(.03)


def test_source_overlap_is_rejected():
    a,meta,cfg = packet(); meta['source_roles']['producer_sites'] = ['b']
    with pytest.raises(AssertionError): diagnostic(a,meta,cfg)


def test_unmatched_query_count_is_rejected():
    a,meta,cfg = packet(); a['nonlinear_matched'][0] = False
    with pytest.raises(AssertionError): diagnostic(a,meta,cfg)


def test_future_field_in_packet_schema_rejected():
    a,meta,cfg = packet(); a['future_endpoint'] = np.zeros((4,2))
    with pytest.raises(ValueError): diagnostic(a,meta,cfg)
