from scripts import report_m3w_scaled_risk as report


def test_interval_preserves_missing_and_signed_values():
    assert 'undefined' in report.interval(dict(mean=None,CI95=None))
    assert report.interval(dict(mean=-1.,CI95=[-2.,-.5]))=='-1.000000 [-2.000000, -0.500000]'


def test_diagnostics_do_not_reclassify_empty_or_negative_cases():
    models={k:dict(dual_cap_reached=True,empty_selection=True,final_TRAIN_monitor=dict(
        soft_all_excess=.1,soft_easy_excess=.001),development_soft_action_mean=.1,
        hard_intervention_rate=0.,soft_at_least_half_hard_disagreement=.1) for k in report.run.heads.NEURAL_ARMS}
    d={k:dict(predicted_observed_easy_harm_ratio=None) for k in report.run.heads.NEURAL_ARMS}
    rows=[dict(training_diagnostics=models,result=dict(diagnostics=d))]*72
    v=report.diagnostics(rows)['budget_scaled']
    assert v['empty_selection_views']==72 and v['TRAIN_soft_easy_excess_views']==72
    assert v['median_predicted_actual_easy_harm'] is None and v['undefined_harm_ratio_views']==72
