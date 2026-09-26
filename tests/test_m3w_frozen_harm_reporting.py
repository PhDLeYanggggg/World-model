import copy
from scripts.report_m3w_european_frozen_harm_readout import contrasts,summarize_contrasts,gates_for,fit_transport


def fixture():
    rows=[]
    for a in range(3):
        for b in range(3):
            if a==b: continue
            for seed in (17,29,43):
                folds=[]
                for held in ('a','b','c','d'):
                    metrics={}
                    for arm,mse in (('original_mean',4.),('mean_features',2.),('fractional_features',1.)):
                        m=dict(harm_MSE=mse,harm_coverage=1.,component_MSE=[1,mse,1,mse],
                            scores=dict(moment=dict(top10_harm_mass_share=.3,AUROC=.7)))
                        metrics[arm]={s:copy.deepcopy(m) for s in ('all','envelope_positive')}
                    folds.append(dict(held=held,metrics=metrics))
                rows.append(dict(pair='full',producer=a,controller=b,seed=seed,folds=folds))
    cfg=dict(pairs=['full'],seeds=[17,29,43],bootstrap_seed=47131,bootstrap_resamples=3000)
    return rows,cfg


def test_two_primary_comparators_required():
    rows,cfg=fixture(); summary=summarize_contrasts(contrasts(rows,cfg))
    assert gates_for(summary)['development_advance_gate']
    assert summary['fractional_vs_matched']['full']['envelope_positive__harm_MSE_gain_percent']['point_range']==[50.,50.]
    for r in rows:
        for f in r['folds']:
            for s in ('all','envelope_positive'): f['metrics']['original_mean'][s]['harm_MSE']=.5
    gates=gates_for(summarize_contrasts(contrasts(rows,cfg)))
    assert gates['matched_primary_six_positive'] and not gates['original_primary_six_positive']
    assert not gates['development_advance_gate']


def test_missing_tail_support_cannot_pass_gate():
    rows,cfg=fixture()
    rows[0]['folds'][0]['metrics']['fractional_features']['envelope_positive']['scores']['moment']['top10_harm_mass_share']=None
    g=gates_for(summarize_contrasts(contrasts(rows,cfg)))
    assert not g['tail_coverage_guards'] and not g['development_advance_gate']


def test_fit_transport_keeps_fit_only_and_missing_support_separate():
    rows,cfg=fixture(); rows=rows[:1]; rows[0]['folds']=rows[0]['folds'][:3]
    for f in rows[0]['folds']:
        f['training']={a:dict(training=dict(harm_MSE=m['all']['harm_MSE'])) for a,m in f['metrics'].items()}
    rows[0]['folds'][0]['metrics']['fractional_features']['all']['harm_MSE']=3.
    rows[0]['folds'][1]['training']['fractional_features']['training']['harm_MSE']=3.
    rows[0]['folds'][2]['training']['fractional_features']['training']['harm_MSE']=None
    c=fit_transport(rows)['fractional_vs_matched']['full']
    assert c==dict(views=3,not_estimable=1,fit_improved=1,held_improved=1,both_improved=0,fit_only=1,held_only=1,neither=0)
