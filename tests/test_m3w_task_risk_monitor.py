import pytest

from scripts import diagnose_m3w_task_risk_monitor as api


def test_monitor_violation_count_is_not_a_calibration_guarantee():
    ds=[dict(arm=arm,trace=[dict(step=2000,monitor=dict(soft_all_excess=-.1,soft_easy_excess=.001))],
             dual=[7.,1.]) for arm in ('quadratic','task_utility','task_risk') for _ in range(72)]
    r=api.summarize(ds)
    assert r['task_risk']['positive_soft_easy_excess']==72
    assert r['task_risk']['positive_soft_all_excess']==0
    assert r['task_risk']['cap_reached_views']==0
    ds.pop()
    with pytest.raises(ValueError):api.summarize(ds)
