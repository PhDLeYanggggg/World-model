from src.world_model import m3w_calibration_size as m


def test_two_disjoint_calibration_sets_never_share_fit_or_held_source():
    names = list('abcdefghijklmn')
    for held in names:
        fit,a,b = m.roles(names,held)
        assert len(fit)==7 and len(a)==len(b)==3
        assert len(set((held,*fit,*a,*b)))==14


def test_gain_pooling_and_constraint_pooling_are_separate():
    def v(pp,ok=True):
        return dict(lower_pp=pp,preserved=ok)
    a = [v(0),v(.5),v(.1),v(.2)]
    b = [v(0),v(.3,False),v(.8),v(.1)]
    chosen,full = m.comparisons(a,b)
    assert chosen==[1,2,2,2,2]
    assert not full[1]['preserved'] and full[1]['lower_pp']==.4
    assert m.mean_pair(dict(reference=1,known_delta=2,unknown_radius=1),
                       dict(reference=1,known_delta=4,unknown_radius=3))==dict(
                           reference=1,known_delta=3,unknown_radius=2)
