from scripts import report_m3w_moving_supported_global_motion as r


def test_primary_na_or_crossing_interval_is_not_positive():
    def value(ci):return {'contrasts':{'primary':{'completion_lower_pp':{'interval95':ci}}}}
    assert not r.primary_positive(value(None))
    assert not r.primary_positive(value([-.1,.4]))
    assert not r.primary_positive(value([0,.4]))
    assert r.primary_positive(value([.01,.4]))


def test_tail_keeps_required_record_view_and_empty_stays_na():
    def row(arm,values):
        return {'arm':arm,'policies':{'joint_free':{'recordings':{
            str(i):{'known_easy_net_degradation':v} for i,v in enumerate(values)}}}}
    a={'rows':[row('x',[None,.5]),row('x',[-.2,.03]),row('y',[9.]) ]}
    assert r.worst_signed(a,'x','joint_free')==.5
    assert r.worst_signed({'rows':[row('x',[None])]},'x','joint_free') is None
    assert r.number(None)=='NA'
