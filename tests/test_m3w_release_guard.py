import numpy as np

from src.world_model import m3w_release_guard as m


def test_every_outer_source_has_disjoint_roles_and_three_calibration_views():
    sources = list('abcdefghijklmn')
    use = {s:0 for s in sources}
    for held in sources:
        fit,cal = m.roles(sources,held)
        assert len(fit)==10 and len(cal)==3
        assert not set(fit)&set(cal) and held not in (*fit,*cal)
        for s in cal:
            use[s]+=1
    assert set(use.values())=={3}


def test_calibration_chooses_only_positive_preserved_candidate():
    choices = [dict(preserved=True,lower_pp=0.),dict(preserved=True,lower_pp=.1),
               dict(preserved=False,lower_pp=.3),dict(preserved=True,lower_pp=None)]
    assert m.choose_threshold(choices)==1
    assert m.choose_threshold(choices[2:])==0


def test_guard_only_vetoes_reopened_actions_and_keeps_original_risk_support():
    r = np.array([[True,False],[True,True]])
    s = np.array([[.01,0.],[.4,.02]])
    np.testing.assert_array_equal(m.keep('learned',2,s,r),[[True,False],[False,True]])
    np.testing.assert_array_equal(m.keep('radius',7,s,r),r)
    old = dict(known_easy_net_degradation=.012,completion_easy_risk_upper=.03,supported=False)
    assert m.preserves(old,dict(old,known_easy_net_degradation=.019))
    assert not m.preserves(old,dict(old,known_easy_net_degradation=.021))
    assert not m.preserves(old,dict(old,completion_easy_risk_upper=None))
