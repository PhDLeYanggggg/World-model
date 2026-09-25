from scripts.report_m3w_european_protected_motion import compact_metric, summary_metrics


def metric(value=None):
    return dict(indexed_rows=4, supported_rows=3, unknown_rows=1,
                equal_scene_gain_percent=value, worst_scene_gain_percent=value,
                scene_bootstrap_ci95=None, by_scene={'held': {'rows': 3}})


def test_summary_keeps_unknown_support_and_undefined_contrasts():
    got = compact_metric(metric(), scenes=True)
    assert got['unknown_rows'] == 1
    assert got['equal_scene_gain_percent'] is None
    assert got['scene_bootstrap_ci95'] is None
    assert got['by_scene'] == {'held': {'rows': 3}}


def test_summary_does_not_confuse_neural_head_with_neural_candidate():
    description = dict(ADE_vs_CV=metric(1), zero_CV={'rows': 0, 'harmed_rows': 0})
    policy = dict(full=description, joint_population={'joint': description},
                  comparisons={'matched': metric()}, queries={'total': 1})
    r = dict(result_source='fixture', source_rows=4, joint_rows=2, new_heads=45,
             cached_heads=45, new_neural_updates=54000,
             policies={'17_neural_easy_neural_underharm4_no_guard': policy,
                       '17_damping097_easy_neural_underharm4_no_guard': policy},
             references={}, neural_vs_damping={}, old_neural_decision_changes={})
    got = summary_metrics(r)
    assert got['candidate_counts'] == {'neural': 1, 'damping097': 1}
    assert len(got['policies']) == 2
    assert got['calibrated_safety'] is False


def test_compact_metric_does_not_change_source_metric():
    original = metric(0)
    got = compact_metric(original)
    assert 'by_scene' not in got
    assert original['by_scene'] == {'held': {'rows': 3}}
    assert got['equal_scene_gain_percent'] == 0
