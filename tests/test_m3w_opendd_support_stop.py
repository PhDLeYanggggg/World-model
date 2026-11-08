from src.world_model import m3w_opendd_support_stop as m


def test_no_end_of_recording_release():
    seq = [(None, 0., 0.), (4, .2, -.1), (None, 0., 0.), (7, -.1, -.1), (8, .4, .4)]
    assert m.stop(iter(seq[:2]), "continue") is None
    assert m.stop(iter(seq[:4]), "continue") == (3, 7)
    assert m.stop(iter(seq), "continue") == (3, 7)
    assert m.stop(iter(seq), "myopic") == (1, 4)


def test_expiry_is_query_based_and_training_targets_are_bounded():
    seq = [(3, .1, -.2), (None, 0., 0.), (None, 0., 0.), (6, 1., 1.)]
    assert m.stop(iter(seq), "wait_one") is None
    assert list(m.targets([None, .25, .5, None, 9.])) == [(1, .25, -.25), (2, .5, .5)]
