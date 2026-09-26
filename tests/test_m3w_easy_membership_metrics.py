import numpy as np
from src.evaluation.m3w_easy_membership_metrics import summarize,paired,gates


def test_binary_metrics_and_fixed_bins_unknown_ignored():
    p=np.array([0.,.25,.75,1.,.99]); y=np.array([0.,0.,1.,1.,np.nan]); s=np.repeat('a',5)
    m=summarize(p,y,s); assert m['rows']==4 and m['positive']==2
    assert m['AUROC']==1 and m['AUPRC']==1
    assert m['Brier']==.03125 and m['ECE']==.125
    assert m['bins'][-1]['rows']==1
    assert summarize(p,y,s,subset=np.zeros(5,bool))['status']=='not_estimable'


def test_single_class_not_reported_as_discrimination_success():
    m=summarize(np.ones(5)*.2,np.zeros(5),np.repeat('a',5))
    assert m['AUROC'] is None and m['AUPRC'] is None and not m['stable_support']


def test_three_seed_locality_aggregation_and_missing_support_gate():
    rows=[]
    for a in range(3):
        for b in range(3):
            if a==b: continue
            for seed in (17,29,43):
                fs=[]
                for held in ('a','b','c','d'):
                    metrics={}
                    for arm in ('linear','mlp','train_constant','reference_ratio'):
                        m=dict(Brier=.1 if arm in ('linear','mlp') else .2,AUROC=.8 if arm in ('linear','mlp') else .5,
                            log_loss=.3 if arm in ('linear','mlp') else .5,ECE=.05)
                        metrics[arm]={s:dict(m) for s in ('all','envelope_positive')}
                    fs.append(dict(held=held,metrics=metrics))
                rows.append(dict(pair='full',producer=a,controller=b,seed=seed,folds=fs))
    c=dict(pairs=['full'],arms=['linear','mlp'],seeds=[17,29,43],bootstrap_seed=47131,bootstrap_resamples=3000)
    r=paired(rows,c); assert gates(r)['membership_signal_established']
    assert r['full']['producer0_controller1']['mlp__envelope_positive__Brier_skill_percent']['CI']==[50.,50.]
    rows[0]['folds'][0]['metrics']['mlp']['envelope_positive']['AUROC']=None
    assert not gates(paired(rows,c))['membership_signal_established']
