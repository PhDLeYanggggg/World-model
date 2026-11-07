import numpy as np

from src.world_model import m3w_context_calibration as m


def test_profile_is_invariant_to_translation_scale_and_agent_order():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4,8,2)).cumsum(1)
    np.testing.assert_allclose(m.profile(x),m.profile(7*x[::-1]+300.),atol=1e-12)
    assert np.isfinite(m.profile(np.zeros((1,8,2)))).all()


def test_nearest_and_farthest_use_one_recording_from_every_source():
    p = {k:np.asarray([v]) for k,v in zip('abcd',[0.,1.,2.,3.])}
    groups = {'first':('a','b'),'second':('c','d')}
    args = (p['a'],p,groups,np.ones(1))
    assert [x for x,_ in m.peers(*args,'near')]==['a','c']
    assert [x for x,_ in m.peers(*args,'far')]==['b','d']
    assert len(m.peers(*args,'blind'))==2


def test_threshold_preserves_all_chosen_recordings():
    def row(v,ok=True):
        return dict(lower_pp=v,preserved=ok)
    assert m.threshold([[row(0.),row(.2),row(.1)],[row(0.),row(.3,False),row(.1)]])==2
