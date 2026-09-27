import numpy as np
from scripts import report_m3w_european_cost_mass as r


def metric(value):
    row = dict(harm_MSE=value,harm_coverage=1.,scores=dict(moment=dict(top10_harm_mass_share=.5,AUROC=.6)))
    return dict(all=dict(row,component_MSE=[1.,1.,1.,value]),envelope_positive=dict(row))


def fixture():
    cfg = dict(pairs=['full'],seeds=[17,29,43],bootstrap_resamples=100,bootstrap_seed=71429)
    values = {arm+'_'+mode:metric(value) for arm in ('cost_only','cap_aux','shuffled_aux')
        for mode,value in [('mass',1.),('raw',2.),('scaled',3.)]}
    rows = [dict(producer=a,controller=b,seed=seed,pair='full',
        folds=[dict(held=s,metrics=values.copy()) for s in 'abcd'])
        for a in range(3) for b in range(3) if a != b for seed in cfg['seeds']]
    return cfg,rows


def test_registered_contrasts_preserve_seed_locality_units():
    cfg,rows = fixture(); out = r.aggregate(rows,cfg,True)
    m = out['contrasts']['mass_cost_only_vs_raw_cost_only']['full']['producer0_controller1'][r.PRIMARY]
    assert m['point'] == 50. and m['CI'] == [50.,50.]
    assert m['localities'] == 4 and m['seeds'] == 3
    assert out['gates']['development_readout_screen']
    assert not out['gates']['auxiliary_information_gate'] and not out['gates']['deployment_changed']


def test_infeasibility_and_missing_support_cannot_pass():
    cfg,rows = fixture(); assert not r.aggregate(rows,cfg,False)['gates']['development_readout_screen']
    rows[0]['folds'][0]['metrics']['cost_only_mass'] = dict(
        all=metric(1.)['all'],envelope_positive=dict(status='not_estimable'))
    out = r.aggregate(rows,cfg,True)
    assert not out['gates']['development_readout_screen']
    assert out['summary']['mass_cost_only_vs_raw_cost_only']['full'][r.PRIMARY]['not_estimable'] == 1


def test_descriptive_summary_retains_unavailable_values():
    s = r.summarize([1.,None,np.nan,3.])
    assert s == dict(records=4,not_estimable=2,minimum=1.,median=2.,maximum=3.)
