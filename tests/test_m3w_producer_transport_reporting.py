from scripts.report_m3w_european_producer_transport import compact


def test_report_preserves_undefined_unsafe_and_zero_switch():
    m=dict(indexed_rows=3,supported_rows=0,unknown_rows=3,by_scene={'a':{'rows':0}},
        equal_scene_gain_percent=None,worst_scene_gain_percent=None,scene_bootstrap_ci95=None)
    view=dict(ADE_vs_CV={'all':m},raw_ADE_vs_CV=m,FDE_vs_CV=m,
        zero_CV={'rows':0,'harmed_rows':0},safety_observed_pass=False,switch_rate=0.,
        ledger={'summary':{},'gain_capture_fraction':None})
    a=dict(result_source='fixture',views={'x':view},small_vs_full={
        'x':{'raw_small_vs_full':m,'policy_small_vs_full':{'all':m}}})
    r=compact(a)
    assert r['views']['x']['ADE_vs_CV']['all']['equal_scene_gain_percent'] is None
    assert not r['views']['x']['safety_observed_pass']
    assert r['views']['x']['switch_rate']==0
    assert not r['new_training'] and not r['threshold_refit'] and not r['deployment_changed']
