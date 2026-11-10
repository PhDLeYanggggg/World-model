from io import StringIO

import numpy as np

from src.data_unification.m3w_swiss_roundabout_prefix import pack, plan
from src.data_unification.m3w_swiss_roundabout_support import scan


def test_numeric_reader_never_parses_future_or_whole_track_class():
    def raw(corrupt=False):
        rows = ['track_id,type,lon,lat,time']
        for tick, centi in enumerate(np.rint(np.arange(20)*100/30).astype(int)):
            for agent in (1, 2):
                kind = 'Truck' if corrupt else 'Car'
                lat = 'not-a-coordinate' if corrupt and tick >= 8 else 47.5+.00001*agent
                lon = 'opaque' if corrupt and tick >= 8 else 8.1+.000001*tick
                rows.append(f'{agent},{kind},{lon},{lat},00:00:00.{centi:02}')
        return StringIO('\n'.join(rows))

    segments, _ = scan(raw())
    wanted, meta = plan(segments, 1)
    assert len(wanted) == 16 and meta['current_tick'].tolist() == [7]
    a = pack(raw(), segments, wanted, meta)
    b = pack(raw(True), segments, wanted, meta)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert not {'future', 'vehicle_type'} & a.keys()
    np.testing.assert_array_equal(a['history'][0, -1], np.zeros(2))
