import copy
import pytest
from scripts.report_m3w_european_support_fractional import paired_contrasts


def fixture():
    cfg=dict(pairs=['full'],seeds=[17,29,43],bootstrap_seed=4,bootstrap_resamples=3000)
    rows=[]
    for seed in cfg['seeds']:
        folds=[]
        for site in ('a','b','c','d'):
            metrics={}
            for arm,mse,coverage,capture,auc in (('mean',2.,.5,.2,.6),('fractional',1.,1.,.3,.7)):
                value=dict(harm_MSE=mse,harm_coverage=coverage,component_MSE=[mse]*4,
                    scores=dict(moment=dict(top10_harm_mass_share=capture,AUROC=auc)))
                metrics[arm]={s:copy.deepcopy(value) for s in ('all','envelope_positive')}
            folds.append(dict(held=site,metrics=metrics))
        rows.append(dict(pair='full',producer=0,controller=1,seed=seed,folds=folds))
    return rows,cfg


def test_paired_locality_MSE_bootstrap():
    rows,cfg=fixture(); c=paired_contrasts(rows,cfg)['full']['producer0_controller1']
    assert c['envelope_positive__harm_MSE_gain_percent']['CI']==pytest.approx([50,50])
    assert c['envelope_positive__top10_gain_pp']['point']==pytest.approx(10)
    assert c['envelope_positive__coverage_log_error_reduction']['point']==pytest.approx(.69314718)


def test_missing_locality_cannot_be_silently_dropped():
    rows,cfg=fixture(); rows[0]['folds'][0]['metrics']['fractional']['envelope_positive']['harm_MSE']=None
    c=paired_contrasts(rows,cfg)['full']['producer0_controller1']
    assert c['envelope_positive__harm_MSE_gain_percent']['status']=='not_estimable'
