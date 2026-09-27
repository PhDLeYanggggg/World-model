import numpy as np
from scripts import report_m3w_european_regime_transport as r


def metrics(value):
    row = dict(harm_MSE=value,harm_coverage=1.,scores=dict(moment=dict(top10_harm_mass_share=.5,AUROC=.6)))
    return dict(all=dict(row,component_MSE=[2.,1.,1.,value]),envelope_positive=dict(row))


def row(seed,site,family='full'):
    cells = {c+'_'+mode:metrics(value) for c,value in zip(r.run.method.CELLS,[4.,2.,3.,1.]) for mode in ('raw','scaled')}
    return dict(producer=0,controller=1,pair=family,seed=seed,held=site,
        replicas=[dict(metrics=dict(outer_cut3=cells,inner_cut2_diagnostic=cells)) for _ in range(3)])


def test_factorial_contrasts_and_interaction():
    v = r.view_contrasts(row(17,'a'),'outer_cut3')
    assert v['cut_at_two_raw'][r.PRIMARY] == 50
    assert np.isclose(v['cut_at_three_raw'][r.PRIMARY],200/3)
    assert v['regime_at_cut3_raw'][r.PRIMARY] == 50
    assert v['interaction_raw']['easy_harm_MSE_interaction_pp'] == 0


def test_replica_seed_averaging_not_pseudoreplicated():
    cfg = dict(pairs=['full'],seeds=[17,29,43],bootstrap_seed=71429,bootstrap_resamples=3000)
    rows = [row(seed,site) for seed in cfg['seeds'] for site in 'abcd']
    result = r.aggregate(rows,cfg)
    ci = result['outer_cut3']['contrasts']['cut_at_two_raw']['full']['producer0_controller1'][r.PRIMARY]
    assert ci['localities'] == 4 and ci['seeds'] == 3 and ci['CI'] == [50.,50.]
    assert result['gates']['development_advance_gate'] is False


def test_missing_support_retained():
    cfg = dict(pairs=['full'],seeds=[17,29,43],bootstrap_seed=71429,bootstrap_resamples=50)
    rows = [row(seed,site) for seed in cfg['seeds'] for site in 'abcd']
    rows[0]['replicas'][0]['metrics']['outer_cut3']['two_cut2_raw']['envelope_positive']['status'] = 'not_estimable'
    out = r.aggregate(rows,cfg)
    m = out['outer_cut3']['contrasts']['cut_at_two_raw']['full']['producer0_controller1'][r.PRIMARY]
    assert m['status'] == 'not_estimable'
