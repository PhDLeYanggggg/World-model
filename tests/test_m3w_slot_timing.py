import numpy as np

from src.evaluation import m3w_slot_timing_capacity as capacity
from src.world_model import m3w_slot_timing as model


def test_timing_edges_preserve_radius_and_respect_direction():
    q = np.repeat(np.arange(4), 2)
    peak = np.array([1., .5, 2., 1.5, .25, 1., 2., 1.])
    rows, slots, a = model.candidates(q, peak, np.ones(4), [1], [1.5], 0, 2)
    np.testing.assert_array_equal(rows, [2, 3, 6])
    np.testing.assert_array_equal(slots, [0, 0, 0])
    np.testing.assert_allclose(a*peak[rows], 1.5)


def test_delayed_emission_is_causal_and_cannot_reuse_rows_or_slots():
    cap = np.ones(9, dtype=int)
    eligible = np.ones(9, dtype=bool)
    t = np.array([3, 4, 5, 6, 7, 8])
    rows = np.arange(len(t))
    slots = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([100., -1., .1, 1., 2., 3.])
    release = np.array([4, 6])
    np.testing.assert_array_equal(model.schedule(t, rows, slots, score, release, eligible, cap, 0), [3])
    a = model.schedule(t, rows, slots, score, release, eligible, cap, 2)
    np.testing.assert_array_equal(a, [2])
    score[t >= 6] = 1000.
    b = model.schedule(t, rows, slots, score, release, eligible, cap, 2)
    np.testing.assert_array_equal(a[t[a] < 6], b[t[b] < 6])


def test_oracle_can_only_recover_easy_at_a_payable_query():
    q = np.repeat(np.arange(6), 2)
    cv = np.zeros((12, 12, 2))
    y = np.ones_like(cv)
    y[[0, 10]] *= .05
    valid = np.ones(cv.shape[:2], dtype=bool)
    eligible = np.ones(6, dtype=bool)
    cap = np.ones(6, dtype=int)
    cuts = [.2, .9]

    def check(ix):
        return capacity.assess(cv, cv, y, valid, q, eligible, cap,
                               ix, np.zeros(len(ix), dtype=int), 1, y[ix], cuts)

    assert check([8]) == "no_eligible_easy"
    assert check([0, 8]) == "risk_infeasible"
    assert check([8, 10]) == capacity.POS
