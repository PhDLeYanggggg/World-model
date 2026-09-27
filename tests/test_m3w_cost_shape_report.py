from scripts import report_m3w_european_cost_shape as r
from tests.test_m3w_cost_mass_report import fixture,metric


def make():
    cfg,rows=fixture()
    for row in rows:
        for fold in row['folds']:
            for arm in ('cost_only','cap_aux','shuffled_aux'):
                fold['metrics'][arm+'_shape_mass']=metric(.5)
                fold['metrics'][arm+'_shape_L2']=metric(.75)
    return cfg,rows


def test_primary_and_independent_roles_unchanged():
    cfg,rows=make(); out=r.aggregate(rows,cfg,True)
    v=out['contrasts']['shape_mass_cost_only_vs_raw_cost_only']['full']['producer0_controller1'][r.PRIMARY]
    assert v['point']==75. and v['CI']==[75.,75.] and v['seeds']==3 and v['localities']==4
    assert out['gates']['development_readout_screen']
    assert not out['gates']['auxiliary_information_gate'] and not out['gates']['deployment_changed']


def test_failed_constraint_or_guard_cannot_advance():
    cfg,rows=make(); assert not r.aggregate(rows,cfg,False)['gates']['development_readout_screen']
    for row in rows:
        for f in row['folds']:
            f['metrics']['cost_only_shape_mass']['all']['component_MSE'][1]=10.
    out=r.aggregate(rows,cfg,True)
    assert out['gates']['primary_readout_gate'] and not out['gates']['guard_gate']
    assert not out['gates']['development_readout_screen']


def test_missing_support_is_retained():
    cfg,rows=make(); rows[0]['folds'][0]['metrics']['cost_only_shape_mass']['envelope_positive']={'status':'not_estimable'}
    out=r.aggregate(rows,cfg,True)
    assert not out['gates']['development_readout_screen']
