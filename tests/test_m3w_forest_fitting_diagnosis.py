import numpy as np
from scripts.diagnose_m3w_multiaction_forest_costs import conditional_summary,METRICS
from src.evaluation.m3w_policy_region_shift import compare_regions


def test_undefined_fitting_ratios_are_counted_not_safe_zero():
    def record(value):
        pool=dict(action_views=3,known_action_views=2,unknown_action_views=1,**{k:value for k in METRICS})
        return dict(diagnosis=dict(forest=dict(selected=pool)))
    result=conditional_summary([record(.25),record(None)],'forest','selected')
    assert result['required_record_views']==2 and result['unknown_action_views']==2
    for row in result['ratio'].values():
        assert row['median']==.25 and row['finite_record_views']==1 and row['undefined_record_views']==1
    empty=conditional_summary([record(None)],'forest','selected')
    assert empty['ratio']['easy_harm_prediction_ratio']['median'] is None


def test_cross_estimator_frozen_regions_preserve_known_mass_and_unknowns():
    p=np.tile([1.,.001,10.,3.,.001],(2,7,1))
    q=np.tile([2.,.002,20.,6.,.002],(2,7,1))
    y=np.tile([1.,.1,5.,2.,.1],(2,7,1));y[1]=np.nan
    allowed=np.ones((2,7),bool);fc=np.array([0,-1]);nc=np.array([-1,1]);scale=np.ones(2)
    a=compare_regions(p,y,allowed,fc,nc,scale,tuple(range(7)))
    b=compare_regions(q,y,allowed,nc,fc,scale,tuple(range(7)))
    assert a['selected']['observed_known_mass']==b['frozen_teacher_region']['observed_known_mass']
    assert b['selected']['observed_known_mass']==a['frozen_teacher_region']['observed_known_mass']
    assert a['frozen_teacher_region']['unknown_action_views']==1
    assert a['frozen_teacher_region']['easy_harm_prediction_ratio'] is None
