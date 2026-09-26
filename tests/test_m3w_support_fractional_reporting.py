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


def test_fit_transport_counts_do_not_conflate_training_with_held_gain():
    from scripts.diagnose_m3w_european_support_fractional import fit_transport_summary
    rows,_=fixture()
    for i,row in enumerate(rows):
        row['group']=str(i)
        for fold in row['folds']:
            fold['training']={a:dict(training=dict(harm_MSE=v)) for a,v in (('mean',2.),('fractional',1.))}
    rows[0]['folds'][0]['metrics']['fractional']['envelope_positive']['harm_MSE']=3.
    rows[0]['folds'][1]['training']['mean']['training']['harm_MSE']=None
    value=fit_transport_summary(rows)['full']
    assert value['training_gain']['improved']==11
    assert value['training_gain']['missing']==1
    assert value['held_gain']['improved']==11
    assert value['fitting_improved_held_not']==1
    assert value['fitting_and_held_improved']==10


def test_error_decomposition_keeps_zero_harm_and_scale_partitions():
    import numpy as np
    from src.evaluation.m3w_harm_error_decomposition import error_decomposition
    old=np.ones((4,4)); new=old*2; y=np.zeros((4,4)); y[1,3]=2; y[3]=np.nan
    d=error_decomposition(old,new,y,np.array([1.,3.,0.,4.]),2.)
    assert d['rows']==2 and d['excess_MSE']==pytest.approx(1.)
    assert d['parts']['zero_harm']['excess_MSE_contribution']==pytest.approx(1.5)
    assert d['parts']['positive_harm']['excess_MSE_contribution']==pytest.approx(-.5)
    assert d['parts']['above_training_edge']['excess_MSE_contribution']==pytest.approx(-.5)
