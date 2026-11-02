import numpy as np

from src.evaluation import m3w_selection_cost_transport as api


def fixture():
    p=np.zeros((4,7,5));y=p.copy();allowed=np.zeros((4,7),bool);allowed[:,6]=True
    p[...,2:4]=10.;p[:,6,0]=1.;p[:,6,1]=.01;p[:,6,4]=.01
    y[...,2:4]=2.;y[:,6,0]=.5;y[:,6,1]=.1;y[:,6,4]=.1
    p[1,6,0]=0;p[2,6,1]=1.;p[2,6,0]=2.;y[-1]=np.nan
    choices=np.array([6,-1,-1,6]);return p,y,allowed,choices


def test_signed_risk_accounting_distinguishes_harm_and_reference_errors():
    p,y,allowed,choices=fixture();out=api.diagnose(p,y,allowed,choices,np.ones(4))
    d=out["selected"]["all_excess_decomposition"]
    assert d["harm_underprediction"]>0 and d["reference_inflation"]>0
    assert np.isclose(d["observed_excess"]-d["predicted_excess"],d["harm_underprediction"]+d["reference_inflation"])
    assert out["selected"]["unknown_action_views"]==1


def test_neural_funnel_keeps_unknowns_and_does_not_modify_choices():
    p,y,allowed,choices=fixture();before=choices.copy();out=api.neural_funnel(p,y,allowed,choices)
    assert out["unknown_neural_rows"]==1 and out["useful_known_neural_rows"]==3
    assert [out["buckets"][k]["rows"] for k in ("predicted_nonpositive_gain","predicted_positive_guard_rejected",
        "guard_passed_other_action_chosen","neural_chosen")]==[1,1,0,1]
    np.testing.assert_array_equal(before,choices)


def test_empty_joint_diagnostic_is_not_a_safe_zero_risk():
    p,y,allowed,_=fixture();out=api.selected_summary(p,y,allowed,np.full(4,-1),np.ones(4))
    assert out["observed_all_risk_on_known"] is None and out["observed_easy_risk_on_known"] is None
    assert out["known_action_views"]==0


def test_feature_shift_uses_given_source_statistics_not_new_scene_statistics():
    x=np.zeros((3,7,444));x[0,0,0]=20.;x[1,0,380]=20.
    pr=dict(mean=np.zeros(444),std=np.ones(444),clip=8.)
    out=api.feature_shift(x,pr,[np.array([0,1]),np.array([2])])
    assert out["rows_with_any_clipped_field"]==2
    assert out["query_counts"]["singleton"]==1 and not out["normalization_fitted"]
