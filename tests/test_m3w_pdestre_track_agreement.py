import numpy as np

from scripts import run_m3w_pdestre_track_agreement as run


def box(k, x, score):
    return [k, x, 0., x+4., 4., score, 1.]


def test_consistency_requires_same_partner_and_the_last_frame():
    frames = []
    for j in range(8):
        rapid = [box(0, j, .6), box(1, j+20, .9), box(2, j+40, .8)]
        mobile = [box(10, j, .7), box(20 if j < 4 else 21, j+20, .8)]
        if j < 7:
            mobile.append(box(30, j+40, .8))
        frames.append(dict(boxes=dict(rapid=rapid, mobilenet=mobile)))
    paths = [[i, [], run.ols4(np.array([[j+20*i+2., 2.] for j in range(8)])).tolist()]
             for i in range(3)]
    d = run.proposals(dict(frames=frames, paths={"rapid": paths}))
    assert d["arms"] == dict(all=[0, 1, 2], consistent_agreement=[0],
                             confidence_matched=[1], frame_agreement_matched=[1])
    assert d["fusion"] == {0: paths[0][2]}


def test_partial_and_missing_targets_are_not_dropped():
    future = [{10: True, 20: True} for _ in range(11)]+[{10: True}]
    m = run.availability([0, 1, 2], {0: 10, 1: 20}, future)
    assert m == dict(kept=3, stable=2, complete=1, absent=1, partial=1,
                     full_selected=False, complete_pair=False)
    assert not run.availability([], {}, future)["full_selected"]
