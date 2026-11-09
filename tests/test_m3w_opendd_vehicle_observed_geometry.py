import sqlite3

import numpy as np

from scripts.build_m3w_opendd_vehicle_observed_geometry import observed, source


def test_observed_join_preserves_roster_offsets_without_future_coordinates():
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE rdb1_1 (OBJID, TIMESTAMP, CLASS, UTM_X, UTM_Y)')
        stride = source.indices.audit.STRIDE
        dt = 1001./30000.
        rows = [(a, k*dt*stride, 'Car', a*10+k if k < 8 else 'poison', 0.)
                for a in (1, 2) for k in range(20)]
        db.executemany('INSERT INTO rdb1_1 VALUES (?, ?, ?, ?, ?)', rows)
        agent, count, centered, offsets = observed(db, 'rdb1_1')
    np.testing.assert_array_equal(agent, [1, 2])
    np.testing.assert_array_equal(count, [2])
    np.testing.assert_array_equal(centered[:, :, 0], np.broadcast_to(np.arange(-7, 1), (2, 8)))
    np.testing.assert_array_equal(offsets, [[0., 0.], [10., 0.]])
