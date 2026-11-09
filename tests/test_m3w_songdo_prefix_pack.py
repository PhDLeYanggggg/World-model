import io
import zipfile

import numpy as np
import pytest

from scripts import build_m3w_songdo_prefix_pack as pack


def test_past_defined_roster_keeps_agent_with_unavailable_future():
    ids = np.repeat([1, 2, 3], [20, 20, 8])
    frame = np.r_[np.arange(20), np.arange(20), np.arange(8)] * 12
    a, s, grid, idx = pack.exact_indices(ids, frame, np.arange(len(ids)), int(frame.max()))
    np.testing.assert_array_equal(a, [1, 2, 3])
    np.testing.assert_array_equal(s, [0, 0, 0])
    np.testing.assert_array_equal(grid[0], np.arange(20)*12)
    assert (idx[:, :8] >= 0).all()
    assert (idx[2, 8:] < 0).all()


def test_sparse_join_rejects_duplicate_identity_frame():
    with pytest.raises(ValueError, match='Duplicate'):
        pack.exact_indices(np.array([1, 1]), np.array([0, 0]), np.array([0, 1]), 0)


def test_coordinate_parser_ignores_unselected_future_xy():
    source = ('Vehicle_ID,Local_Time,Drone_ID,Vehicle_Class,Visibility,Local_X,Local_Y\n'
              '1,07:00:00.000,1,0,1,12.0,13.0\n'
              '1,07:00:00.400,1,0,1,not_numeric,do_not_parse\n'
              '2,07:00:00.400,1,0,1,22.0,23.0\n')
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w') as archive:
        archive.writestr('fixture.csv', source)
    with zipfile.ZipFile(blob) as archive:
        d = pack.audit.load(archive, 'fixture.csv')
        a = pack.past_coordinates(archive, 'fixture.csv', d, np.array([[0, 2]]))
        np.testing.assert_array_equal(a, [[[12., 13.], [22., 23.]]])
        with pytest.raises(ValueError):
            pack.past_coordinates(archive, 'fixture.csv', d, np.array([[1]]))


def test_pack_inputs_unchanged_when_one_actors_future_is_removed():
    name = 'fixture.csv'

    def make(missing):
        rows = ['Vehicle_ID,Local_Time,Drone_ID,Vehicle_Class,Visibility,Local_X,Local_Y']
        for agent in (1, 2):
            for step in range(8 if missing and agent == 2 else 20):
                ms = int(round(step * 12 * pack.audit.DT_MS))
                x = str(step) if step < 8 else 'never_read'
                y = str(agent) if step < 8 else 'never_read'
                rows.append(f'{agent},07:00:{ms//1000:02}.{ms%1000:03},1,0,1,{x},{y}')
        blob = io.BytesIO()
        with zipfile.ZipFile(blob, 'w') as z:
            z.writestr(name, '\n'.join(rows) + '\n')
        with zipfile.ZipFile(blob) as z:
            receipt = pack.audit.session(pack.audit.load(z, name), name, 'A')
            return pack.arrays(z, receipt)

    a, labels = make(False)
    b, missing = make(True)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert labels['future_mask'].all()
    assert missing['future_mask'][0].all()
    assert not missing['future_mask'][1].any()


def test_linkage_can_find_a_nearby_track_with_different_camera_clock_phase():
    from scripts import audit_m3w_songdo_prefix_linkage as audit

    a_time = np.arange(8) * 400
    b_time = a_time + 30
    a = np.column_stack((a_time / 100, np.zeros(8)))
    b = np.column_stack((b_time / 100, np.ones(8)))
    assert audit.close_during_overlap(a, a_time, b, b_time)
    assert not audit.close_during_overlap(a, a_time, b, b_time + 2400)
    assert not audit.close_during_overlap(a, a_time, b + [0, 5], b_time)
    r = audit.compare(a[None], a_time[None], np.stack((b, b)), np.stack((b_time, b_time+4000)))
    assert r['possible_duplicate_prefix_pairs'] == 1
