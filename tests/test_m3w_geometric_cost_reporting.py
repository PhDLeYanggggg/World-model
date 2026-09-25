from scripts.report_m3w_european_geometric_cost import summary


def test_unknown_unsafe_and_negative_results_are_not_dropped():
    m=dict(indexed_rows=5,supported_rows=0,unknown_rows=5,by_scene={},
        equal_scene_gain_percent=None,worst_scene_gain_percent=None,scene_bootstrap_ci95=None)
    view=dict(ADE_vs_CV={'all':m},FDE_vs_CV=m,raw_ADE_vs_CV=m,
        zero_CV={'rows':1,'harmed_rows':1},decision_sha256='fixture',safety_observed_pass=False,
        switch_rate=0.,ledger={'summary':{},'gain_capture_fraction':None})
    a=dict(result_source='fixture',views={'x':view},replacements={'x':{'all':m}},neural_vs_damping={})
    r=summary(a)
    assert r['views']['x']['ADE_vs_CV']['all']['unknown_rows']==5
    assert r['views']['x']['ADE_vs_CV']['all']['equal_scene_gain_percent'] is None
    assert not r['views']['x']['safety_observed_pass']
    assert r['views']['x']['zero_CV']['harmed_rows']==1
    assert not r['deployment_changed'] and not r['forecast_training']
