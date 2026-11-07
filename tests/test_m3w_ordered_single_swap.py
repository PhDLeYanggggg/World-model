import numpy as np

from scripts.audit_m3w_ordered_single_swap import witnesses


def test_later_removal_requires_space_when_the_insertion_arrives():
    pool = dict(query=np.array([1]), selected_query=np.array([0, 2]),
                cap=np.array([1, 0, 1]), gain=np.array([1.]),
                deletion_gain=np.array([1., 1.]), selected_cost=np.zeros((3, 2)),
                cost=np.zeros((3, 1)), full_easy=1.)
    assert witnesses(pool) == {}
    pool['cap'][1] = 1
    assert witnesses(pool) == {'later_query': (0, 1)}
    pool['query'][0] = 2
    assert witnesses(pool) == {'same_query': (0, 1)}


def test_ordered_pair_must_be_nonloss_and_respect_easy_risk():
    pool = dict(query=np.array([0]), selected_query=np.array([1]),
                cap=np.array([1, 1]), gain=np.array([0.]),
                deletion_gain=np.array([-.1]), selected_cost=np.zeros((3, 1)),
                cost=np.array([[0.], [0.], [.02]]), full_easy=1.)
    assert witnesses(pool) == {}
    pool['gain'][0] = .1
    assert witnesses(pool) == {'later_query': (0, 0)}
    pool['cost'][2, 0] = .021
    assert witnesses(pool) == {}
