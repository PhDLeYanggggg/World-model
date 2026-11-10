from io import StringIO

import numpy as np
import pytest

from src.data_unification.m3w_pneuma_prefix import east_north, pack, plan
from src.data_unification.m3w_pneuma_support import HEADER
from src.evaluation.m3w_pneuma_future_join import join


def source(missing=True):
    table = [';'.join(HEADER)]
    for a in (1, 4):
        row = [str(a), 'Car', 'unused', 'unused']
        for t in range(20):
            lat = '' if missing and (a, t) == (4, 9) else f'{38.+a*.0001+t*.00001:.6f}'
            row += [lat, '23.75', 'unused', 'unused', 'unused', f'{.04*t:.6f}']
        table.append(';'.join(row))
    return '\n'.join(table)


def test_join_aligns_actor_grid_and_last_observation_origin():
    txt = source()
    pre = pack(StringIO(txt), *plan(StringIO(txt), 1), 1)
    y, valid = join(StringIO(txt), pre, 1)
    assert y.shape == (2, 12, 2)
    assert not valid[1, 1] and valid.sum() == 23
    assert np.all(y[~valid] == 0.)
    expected = east_north([38.00048, 23.75])-east_north([38.00047, 23.75])
    np.testing.assert_allclose(y[1, 0], expected, rtol=0., atol=1e-10)
    np.testing.assert_array_equal(pre['agent_id'], [1, 4])


def test_join_fails_when_a_frozen_past_actor_disappears():
    txt = source(False)
    pre = pack(StringIO(txt), *plan(StringIO(txt), 1), 1)
    with pytest.raises(ValueError, match='actor absent'):
        join(StringIO('\n'.join(txt.splitlines()[:2])), pre, 1)
