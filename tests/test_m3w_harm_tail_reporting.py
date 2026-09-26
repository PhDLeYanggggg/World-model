import numpy as np
import pytest
from scripts.report_m3w_european_harm_tail_crossfit import ci_by_assignment
from scripts.verify_m3w_european_harm_tail_crossfit import rank_metrics,top_share


def fixture():
    cfg=dict(pairs=['full'],seeds=[17,29,43],bootstrap_seed=7,bootstrap_resamples=3000)
    rows=[]
    for seed in cfg['seeds']:
        folds=[]
        for site in ['a','b','c','d']:
            scores={k:dict(top10_harm_mass_share=v) for k,v in [('moment',.6),('fraction',.5),('envelope',.4)]}
            folds.append(dict(held=site,held_metrics={s:dict(scores=scores) for s in ('all','envelope_positive')}))
        rows.append(dict(pair='full',producer=0,controller=1,seed=seed,folds=folds))
    return cfg,rows


def test_equal_locality_seed_average_not_window_bootstrap():
    cfg,rows=fixture(); v=ci_by_assignment(rows,cfg)['full']['producer0_controller1']['all__moment_vs_envelope']
    assert v['point_pp']==pytest.approx(20) and v['CI_pp']==pytest.approx([20,20])
    assert v['localities']==4 and v['seeds']==3


def test_missing_event_support_is_not_silently_dropped():
    cfg,rows=fixture(); rows[0]['folds'][0]['held_metrics']['all']['scores']['moment']['top10_harm_mass_share']=None
    assert ci_by_assignment(rows,cfg)['full']['producer0_controller1']['all__moment_vs_envelope']['status']=='not_estimable'


def test_independent_ranking_and_mass_ties():
    score=np.array([0.,1.,1.,2.]); harm=np.array([0.,3.,0.,1.])
    auc,ap=rank_metrics(score,harm)
    assert auc==pytest.approx(.875) and ap==pytest.approx(5/6)
    assert top_share(np.ones(4),harm,.1)==pytest.approx(.1)
    assert rank_metrics(score,np.zeros(4))==(None,None)
