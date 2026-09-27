import numpy as np
import pytest
from src.world_model.m3w_observation_quality import (
    history_diagnostics, masked_neighbors, repair_geometry, prefix_continuity)
from src.world_model.m3w_european_source_forecast import pack_scene
from src.data_unification.m3w_european_squares_source import build_recording
from src.evaluation.m3w_european_squares_intake import DTYPE


def source():
    r = np.zeros(22, dtype=DTYPE)
    r['agent'][:20] = 1; r['frame'][:20] = np.arange(20)*12
    r['agent'][20:] = 2; r['frame'][20:] = [72, 84]
    r['x_min'] = r['frame']*.1; r['x_max'] = r['x_min']+4
    r['y_min'] = r['agent']*2; r['y_max'] = r['y_min']+8
    r['confidence'] = .9
    return r


def scene(r):
    data, _ = build_recording(r, [84])
    return {k:v for k,v in data.items() if k != 'query_offsets'}


def test_newcomer_has_tokens_and_missing_entries_are_zero():
    s = scene(source()); old, ids = pack_scene(s); n = masked_neighbors(s)
    np.testing.assert_array_equal(ids, n['target'])
    assert old[:, 230:294].sum() == 0
    assert n['valid'].sum() == 2 and n['partial_neighbors'][0] == 1
    assert n['agent_ids'][0, 0] == 2
    new = repair_geometry(old, n)
    np.testing.assert_array_equal(new[:, :38], old[:, :38])
    np.testing.assert_array_equal(new[:, 294:], old[:, 294:])
    assert not n['xy'][~n['valid']].any()
    poisoned = dict(s); poisoned['history_xy'] = s['history_xy'].copy()
    poisoned['history_xy'][~s['history_valid']] = np.nan
    np.testing.assert_array_equal(masked_neighbors(poisoned)['xy'], n['xy'])


def test_future_mutation_and_truncation_invariance():
    r = source(); q = r.copy()
    for f in ('x_min', 'x_max', 'y_min', 'y_max'):
        q[f][q['frame'] > 84] += 999
    expected = masked_neighbors(scene(r))
    for modified in (q, r[r['frame'] <= 84]):
        result = masked_neighbors(scene(modified))
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])
    invalid = scene(r); invalid['future_xy'] = np.zeros((2, 12, 2))
    with pytest.raises(ValueError): masked_neighbors(invalid)


def test_complete_neighbors_match_legacy_and_order_invariant():
    s = scene(source()); s['history_xy'][1] = s['history_xy'][0]+[1, 2]
    s['history_valid'][:] = True; s['target_eligible'][:] = True
    old, _ = pack_scene(s)
    np.testing.assert_array_equal(repair_geometry(old, masked_neighbors(s)), old)
    reversed_s = {k:v[::-1] for k,v in s.items()}
    result = masked_neighbors(reversed_s)
    np.testing.assert_array_equal(result['xy'][::-1], masked_neighbors(s)['xy'])


def test_fixed_motion_and_pseudo_future_are_inside_prefix():
    h = np.stack([np.arange(8), 2*np.arange(8)], axis=1)[None].astype(float)
    b = np.concatenate((h-[2, 4], h+[2, 4]), axis=2)
    d = history_diagnostics(h, b)
    for k in ('line_residual_over_width', 'last_fd_ols8_disagreement_over_width',
              'observed_prefix_fd_error', 'observed_prefix_ols4_error', 'observed_prefix_ols6_error'):
        np.testing.assert_allclose(d[k], 0, atol=1e-14)
    for value in history_diagnostics(h+[123, 456], b+[123, 456, 123, 456]).values():
        assert np.isfinite(value).all()
    broken = b.copy(); broken[..., 2] = broken[..., 0]-1
    with pytest.raises(ValueError): history_diagnostics(h, broken)


def test_continuity_never_uses_future_and_rejects_duplicate():
    r = source()[:20]; a = prefix_continuity(r, [84])
    np.testing.assert_array_equal(a, prefix_continuity(r[r['frame'] <= 84], [84]))
    assert a[0, 0] == 8/85 and a[0, 1] == 12
    with pytest.raises(ValueError): prefix_continuity(r, [85])
    r['frame'][1] = r['frame'][0]
    with pytest.raises(ValueError): prefix_continuity(r, [84])
