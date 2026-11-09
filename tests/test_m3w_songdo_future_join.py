import numpy as np
import pytest

from src.evaluation.m3w_songdo_future_join import displacements


def example():
    history = np.zeros((2, 8, 2))
    history[1, -1] = [4., 0.]
    p = dict(history=history, offsets=np.array([0, 2]), past_ms=np.zeros((2, 8)),
             agent_id=np.array([19, 23]), drone=np.array([2]))
    rows = np.full((2, 12), -1)
    rows[0, 0], rows[1, 1] = 41, 39
    ix = dict(future_index=rows, future_mask=rows>=0, origin=np.array([[120., 200.]]))
    s = dict(row=np.array([39, 41]), xy=np.array([[126., 200.], [121., 200.]]),
             agent=np.array([23, 19]), drone=np.array([2, 2]), ms=np.array([801., 400.]),
             visible=np.ones(2, int))
    return p, ix, s


def test_join_recovers_per_actor_displacement_instead_of_shared_origin():
    p, ix, s = example()
    out = displacements(p, ix, s)
    np.testing.assert_array_equal(out[ix['future_mask']], [[1., 0.], [2., 0.]])
    assert np.isnan(out[~ix['future_mask']]).all()


@pytest.mark.parametrize('key,new', [('agent', 2), ('drone', 9), ('ms', 400.), ('visible', 0)])
def test_join_rejects_an_out_of_order_track_or_clock(key, new):
    p, ix, s = example()
    s[key][0] = new
    with pytest.raises(ValueError, match='does not match the prefix'):
        displacements(p, ix, s)
