import copy
import pytest
from scripts.report_m3w_european_cost_attribution import aggregate


def rows():
    d=dict(easy=dict(MSE=2.,gain_vs_original_percent=-10.,label_assisted_E_gain_vs_original_percent=40.,
        label_assisted_E_gain_vs_composed_percent=50.,label_assisted_H_gain_vs_original_percent=30.,
        membership_squared=4.,severity_squared=1.,cross_term=-3.),
        all_harm=dict(label_assisted_E_gain_vs_original_percent=5.),membership_Brier=.1,severity_weighted_Brier=.2,
        strata={k:dict(easy_MSE_share=v) for k,v in (('outside_easy',.6),('outside_easy_low_probability',.1),('outside_easy_high_severity',.3))})
    return [dict(pair=p,producer=a,controller=b,seed=s,
            folds=[dict(held=str(i),diagnosis=dict(subsets={k:copy.deepcopy(d) for k in ('all','disagreement')})) for i in range(4)])
            for p in ('full','motion_only') for a,b in ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1)) for s in (17,29,43)]


def test_signed_terms_and_all_roles_are_kept():
    result=aggregate(rows()); x=result['summary']['full']['disagreement']
    assert x['composed_gain']['negative']==6 and x['E_assisted_gain']['positive']==6
    assert x['cross_over_MSE']['point_range']==pytest.approx([-1.5,-1.5])
    assert result['label_assisted_not_model_results']


def test_missing_seed_or_support_cannot_be_dropped():
    rs=rows(); rs[0]['folds'][0]['diagnosis']['subsets']['disagreement']={'status':'not_estimable'}
    result=aggregate(rs)
    assert result['summary']['full']['disagreement']['composed_gain']['not_estimable']==1
    with pytest.raises(AssertionError): aggregate(rs[1:])
