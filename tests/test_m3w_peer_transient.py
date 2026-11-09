import numpy as np

from src.world_model.m3w_peer_transient import candidates, replace_recipients


def prefix():
    h = np.zeros((4, 8, 2))
    h[:, :, 0] = np.arange(8.)
    h[:, :, 1] = np.arange(4.)[:, None]
    h[0, -1, 0] += .25
    h[1, -1, 0] -= .25
    q = np.array([0, 0, 0, 1])
    return h, q, cv_path(h)


def cv_path(h):
    return (h[:, -1:] - h[:, -2:-1]) * np.arange(1., 13.)[None, :, None]


def test_peer_changes_shape_but_not_candidate_peak_or_query_membership():
    h, q, cv = prefix()
    paths, peers, cap = candidates(h, q, cv)
    assert peers.tolist() == [1, 2, 1, -1]
    assert not np.allclose(paths[0, 0], paths[0, 1])
    assert cap[0] > 0
    np.testing.assert_allclose(np.linalg.norm(paths - cv[:, None], axis=-1).max(axis=-1),
                               np.repeat(cap[:, None], 2, axis=1))
    np.testing.assert_array_equal(paths[3], np.stack([cv[3], cv[3]]))


def test_translation_rotation_and_scale_equivariance():
    h, q, cv = prefix()
    paths, peers, cap = candidates(h, q, cv)
    rot = np.array([[0., 1.], [-1., 0.]])
    h2 = h @ rot * 3. + np.array([8., 5.])
    p2, j2, c2 = candidates(h2, q, cv_path(h2))
    np.testing.assert_array_equal(peers, j2)
    np.testing.assert_allclose(c2, 3. * cap, atol=1e-12)
    np.testing.assert_allclose(p2, paths @ rot * 3., atol=1e-12)


def test_fixed_recipient_replacement_never_changes_singletons():
    h, q, cv = prefix()
    paths, _, _ = candidates(h, q, cv)
    control = cv.copy()
    control[[0, 1]] += .1
    control[3] += .1
    a, active_a = replace_recipients(cv, control, paths[:, 0], q)
    b, active_b = replace_recipients(cv, control, paths[:, 1], q)
    np.testing.assert_array_equal(active_a, [True, True, False, False])
    np.testing.assert_array_equal(active_a, active_b)
    np.testing.assert_array_equal(a[2:], control[2:])
    np.testing.assert_array_equal(b[2:], control[2:])


def test_summary_accepts_cached_rows_with_an_existing_record_id():
    from scripts import run_m3w_opendd_peer_transient as run

    old = dict(record='old', edits=0, state='no_edits', solve={})
    new = dict(edits=0, state='no_edits', solve={})
    rows = [dict(record=f'site:{i}', menus=dict(zip(run.ARMS, (old, new))),
                 direct=None, new_slot_candidates=0, replacements=0) for i in range(2)]
    assert all(x['records'] == 2 for x in run.summarize(rows)['menus'].values())
