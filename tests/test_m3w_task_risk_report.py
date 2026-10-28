import copy

from scripts import report_m3w_task_risk as api
from tests.test_m3w_task_risk_readout import grid


def test_diagnostics_retain_empty_and_underestimated_views():
    rows,_,_=grid()
    diagnostic=dict(dual=[1.,2.],dual_cap_reached=False,empty_selection=False,
        development_soft_action_mean=.1,hard_intervention_rate=.2,
        soft_at_least_half_hard_disagreement=.05,final_TRAIN_monitor=dict(decision_regret=.3))
    for row in rows:
        row['training_diagnostics']={a:copy.deepcopy(diagnostic) for a in api.run.fitting.ARMS}
        for a in api.run.fitting.ARMS:
            row['result']['diagnostics'][a]['predicted_observed_easy_harm_ratio']=.1
    rows[0]['training_diagnostics']['task_risk']['empty_selection']=True
    rows[0]['result']['diagnostics']['task_risk']['predicted_observed_easy_harm_ratio']=None
    result=api.diagnostics(rows)
    assert result['task_risk']['empty_selection_views']==1
    assert result['task_risk']['undefined_selected_harm_ratio_views']==1
    assert result['task_risk']['selected_easy_harm_underprediction_views']==71
    assert not result['task_risk']['calibrated_probability_claim']


def test_missing_ci_is_not_formatted_as_success():
    assert api.interval(dict(mean=None,CI95=None))=='undefined; support retained'
    assert api.interval(dict(mean=-2.,CI95=[-3.,-1.]))=='-2.000000 [-3.000000, -1.000000]'
