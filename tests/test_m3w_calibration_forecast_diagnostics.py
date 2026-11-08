import numpy as np
import pytest

from src.evaluation.m3w_calibration_forecast_diagnostics import CV, MATCHED, aggregate, individual, relative


def test_mask_and_query_weighting_do_not_treat_missing_endpoint_as_observed():
    y = np.zeros((3,2,2)); y[1,1] = np.nan
    valid = np.array([[True,True],[True,False],[True,True]])
    p = np.zeros_like(y); p[...,0] = [[2,2],[0,1000],[4,4]]
    q = np.zeros_like(p)
    table, counts = individual({CV:p,'joint':q},y,valid,[0,2,3],easy_cut=2.,hard_cut=3.)
    assert table[CV,'all','ade'] == (5.,2)
    assert table[CV,'all','fde'] == (6.,2)
    assert counts == dict(all=dict(ade_rows=3,fde_rows=2),easy=dict(ade_rows=1,fde_rows=1),
                          hard=dict(ade_rows=1,fde_rows=1))


def test_pair_bound_uses_forecast_disagreement_for_unobserved_pair_steps():
    y = np.zeros((2,2,2)); y[0,1] = np.nan
    valid = np.array([[True,False],[True,True]])
    p = np.zeros_like(y); p[0,:,0] = 2.
    paths = {k: p.copy() if k != 'joint' else np.zeros_like(y) for k in MATCHED}
    out, n = relative(paths,y,valid,[np.array([[0],[1]])])
    assert n == (1,2)
    assert out['independent','observed'] == (2.,1.)
    assert out['independent','grid_lower'] == (0.,1.)
    assert out['independent','grid_upper'] == (2.,1.)


def test_paired_sources_not_pooled_and_missing_stays_missing():
    r,cv = aggregate([('x',7,{(CV,'easy','ade'):(10,1),('joint','easy','ade'):(0,1)}),
                      ('y',7,{(CV,'easy','ade'):(0,10),('joint','easy','ade'):(0,10)})],
                     {'x':'A','y':'B'},[7],draws=100)
    assert r[CV,'easy','ade']['point'] == cv['joint','easy','ade']['point'] == 5.
    empty = [('x',7,{(CV,'easy','ade'):(0,0),('joint','easy','ade'):(0,0)})]
    assert aggregate(empty,{'x':'A'},[7],draws=2)[1]['joint','easy','ade']['point'] is None
    with pytest.raises(ValueError,match='Missing or duplicate'):
        aggregate(empty,{'x':'A','y':'B'},[7],draws=2)
