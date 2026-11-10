from io import StringIO

import numpy as np

from src.data_unification.m3w_pneuma_prefix import east_north, ecef, pack, plan, ORIGIN_LAT_LON
from src.data_unification.m3w_pneuma_support import HEADER


def test_ellipsoid_projection_axes_and_order():
    np.testing.assert_allclose(ecef([0., 0.]), [6378137., 0., 0.], atol=1e-10)
    a, b = ORIGIN_LAT_LON
    p = east_north([[a, b], [a, b+.0001], [a+.0001, b]])
    np.testing.assert_allclose(p[0], 0., atol=1e-10)
    np.testing.assert_allclose(p[1], [8.783, 0.], atol=.01)
    np.testing.assert_allclose(p[2], [0., 11.100], atol=.01)


def fixture(future):
    lines = [';'.join(HEADER)]
    for agent in (3, 1):
        row = [str(agent), 'Car', 'summary-not-used', 'summary-not-used']
        for i in range(20):
            position = str(38.+agent*.0001+i*.00001) if i < 8 else future
            row.extend((position, '23.75', 'not-input', 'not-input', 'not-input', f'{i*.04:.6f}'))
        lines.append(';'.join(row))
    return '\n'.join(lines)


def test_prefix_is_identical_with_opaque_or_empty_future_coordinates():
    result = []
    for text in (fixture('not-a-coordinate'), fixture('')):
        wanted, starts, counts = plan(StringIO(text), 1)
        data = pack(StringIO(text), wanted, starts, counts, 1)
        assert data['agent_id'].tolist() == [1, 3]
        assert data['past_tick'].max() == 7
        assert data['offsets'].tolist() == [0, 2]
        assert set(data) == {'history', 'offsets', 'origin', 'start_tick', 'past_tick', 'agent_id', 'vehicle_type'}
        result.append(data['history'])
    np.testing.assert_array_equal(*result)
