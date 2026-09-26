import numpy as np
from scripts.report_m3w_european_easy_membership import conditional_constant_sensitivity


def test_conditional_constant_uses_fitting_prevalence_not_held_prevalence():
    rows=[]
    for seed in (17,29,43):
        fs=[]
        for held in ('a','b','c','d'):
            fs.append(dict(held=held,training={'mlp':{'model':{'envelope_positive':{'positive_rate':.4}}}},
                metrics={'mlp':{'envelope_positive':{'positive_rate':.8,'Brier':.2,'log_loss':.6}}}))
        rows.append(dict(pair='full',producer=0,controller=1,seed=seed,folds=fs))
    cfg=dict(pairs=['full'],arms=['mlp'],bootstrap_seed=47131,bootstrap_resamples=3000)
    r=conditional_constant_sensitivity(rows,cfg)['full']['producer0_controller1']
    np.testing.assert_allclose(r['mlp__Brier_skill_percent']['point'],37.5)
    np.testing.assert_allclose(r['mlp__log_loss_gain']['point'],-(.8*np.log(.4)+.2*np.log(.6))-.6)
