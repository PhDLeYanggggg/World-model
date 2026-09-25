from scripts import report_m3w_european_symmetric_risk as report


def test_summary_preserves_undefined_and_failure_states(monkeypatch):
    monkeypatch.setattr(report,'sha',lambda p:'fixture_hash')
    metric=dict(indexed_rows=4,supported_rows=3,unknown_rows=1,equal_scene_gain_percent=None,
        worst_scene_gain_percent=None,scene_bootstrap_ci95=None,by_scene={'site':{'rows':3}})
    desc=dict(ADE_vs_CV=metric,zero_CV={'rows':0,'harmed_rows':0},safety_observed_pass=False)
    r=dict(result_source='fixture',source_rows=4,joint_rows=3,
        policies={'fixed':dict(full=desc,joint_population={'joint':desc},comparisons={'matched':metric},
            queries={'solver_failures':{'joint':1}})},neural_vs_damping={'view':{'pointwise':metric}},
        symmetric_vs_asymmetric={},decision_changes={})
    got=report.summary(r)
    assert got['new_heads']==36 and got['new_neural_updates']==72000
    assert got['policies']['fixed']['full']['ADE_vs_CV']['equal_scene_gain_percent'] is None
    assert got['policies']['fixed']['full']['safety_observed_pass'] is False
    assert got['policies']['fixed']['queries']['solver_failures']['joint']==1
    assert got['calibrated_safety'] is False and got['deployment_changed'] is False
