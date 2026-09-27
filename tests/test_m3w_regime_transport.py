import numpy as np
import pytest
import torch
from src.world_model import m3w_regime_transport as m
from src.world_model import m3w_membership_auxiliary as neural


def data():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(90,7)).astype(np.float32)
    cv = np.r_[np.linspace(.1,3,30),np.linspace(.2,4,30),np.linspace(5,12,30)]
    raw = np.column_stack([cv,cv/2]); sites = np.repeat(['a','b','c'],30)
    cv[-1] = np.nan; raw[-1] = np.nan
    return x,raw,cv,sites


def test_crossed_cut_changes_only_easy_targets_and_declared_definition():
    x,raw,cv,sites = data()
    bank = {c:m.inputs(x,raw,cv,sites,'d','c',c) for c in m.CELLS}
    assert bank['two_cut2'][1]['positive_easy_cut'] != bank['two_cut3'][1]['positive_easy_cut']
    for prefix in ('two','three'):
        a,b = [bank[prefix+'_cut'+k] for k in ('2','3')]
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[2][:,:2],b[2][:,:2])
        for key in ('mean','std','known','weights'): np.testing.assert_array_equal(a[1][key],b[1][key])
    assert bank['two_cut3'][4]['cut_sites'] == ['a','b','c']
    assert bank['two_cut3'][4]['row_sites'] == ['a','b']
    assert not bank['two_cut3'][4]['eligible_as_inner_OOF']


@pytest.mark.parametrize('cell',m.CELLS)
def test_outer_leakage_rejected(cell):
    with pytest.raises(ValueError): m.inputs(*data(),'c','a',cell)


def test_label_definition_site_cannot_be_claimed_held():
    *_,lineage = m.inputs(*data(),'d','c','two_cut3')
    m.check_prediction_sites(['d'],lineage)
    with pytest.raises(ValueError): m.check_prediction_sites(['c'],lineage)
    with pytest.raises(ValueError): m.check_prediction_sites([],lineage)


def test_same_cut_across_regimes():
    bank = {c:m.inputs(*data(),'d','c',c) for c in m.CELLS}
    for k in ('2','3'):
        assert bank['two_cut'+k][1]['positive_easy_cut'] == bank['three_cut'+k][1]['positive_easy_cut']


def test_native_two_excludes_omitted_values_but_crossed_cut_declares_them():
    x,raw,cv,sites = data(); a = m.inputs(x,raw,cv,sites,'d','c','two_cut2')
    changed = cv.copy(); changed[sites == 'c'] = .01
    b = m.inputs(x,raw,changed,sites,'d','c','two_cut2')
    assert a[1]['positive_easy_cut'] == b[1]['positive_easy_cut']
    c = m.inputs(x,raw,changed,sites,'d','c','two_cut3')
    assert c[1]['positive_easy_cut'] != a[1]['positive_easy_cut']


def test_real_training_crossed_cut_draws_resume(tmp_path):
    torch.set_num_threads(4); x,raw,cv,sites = data()
    settings = dict(steps=10,width=8,batch_size=16,learning_rate=.0003,gradient_clip=5,checkpoint_every=2,heartbeat_every=2)
    def fit(cell,folder,**kw):
        take,pr,y,e,_ = m.inputs(x,raw,cv,sites,'d','c',cell)
        return neural.fit(x[take],y,e,sites[take],np.full(take.sum(),20.),pr,arm='cost_only',seed=17,
            settings=settings,identity=dict(cell=cell),directory=folder,heartbeat=lambda **kw:None,**kw)
    fit('two_cut2',tmp_path/'native'); a,_ = fit('two_cut3',tmp_path/'new')
    fit('two_cut3',tmp_path/'resume',stop_at=4); b,_ = fit('two_cut3',tmp_path/'resume',resume=True)
    for key in a.state_dict(): assert torch.equal(a.state_dict()[key],b.state_dict()[key])
    _,s1 = neural.restore(tmp_path/'native'); _,s2 = neural.restore(tmp_path/'new')
    m.check_cut_match(s1,s2); assert not s2['draws'][~s2['preprocess']['known']].any()


def test_missing_replica_never_dropped():
    assert m.mean_replica_deltas([dict(a=1),dict(a=2),dict(a=None)]) == dict(a=None)
    with pytest.raises(ValueError): m.mean_replica_deltas([dict(a=1)])
