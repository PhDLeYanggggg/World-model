import numpy as np
import pytest

from scripts.audit_m3w_pneuma_proposal_separation import ARMS, separate


def proposals():
    rows = dict(independent=[1, 0], additive=[0, 2], same_family=[0, 1], joint=[0, 1])
    items = {}
    for arm in ARMS:
        items[f'{arm}_rows'] = np.array([rows[arm]])
        items[f'{arm}_paths'] = np.ones((1, 2, 12, 2))
        items[f'{arm}_paths'][..., 1] = 0.
    items['same_family_paths'][0, 0, :-1, 0] = -1.
    return items


def test_two_edit_separation_aligns_identity_and_uses_full_grid():
    out = separate(np.zeros((3, 12, 2)), [0, 3], [1.], proposals())
    assert out['independent']['distinct_paths'] == 0
    assert out['additive']['different_recipients'] == 1
    assert out['additive']['mean_full_grid_ade_envelope'] == pytest.approx(2./3.)
    assert out['same_family']['mean_full_grid_ade_envelope'] == pytest.approx(22./36.)
    assert out['same_family']['mean_full_grid_fde_envelope'] == 0.


def test_unequal_radius_rejected():
    p = proposals()
    p['independent_paths'][0, 1, 0, 0] = 2.
    with pytest.raises(AssertionError):
        separate(np.zeros((3, 12, 2)), [0, 3], [1.], p)


def test_forecast_separation_bounds_paired_error_for_arbitrary_labels():
    rng = np.random.default_rng(10)
    p = proposals()
    cv = np.zeros((3, 12, 2))
    bounds = separate(cv, [0, 3], [1.], p)
    a, b = cv.copy(), cv.copy()
    a[p['joint_rows'][0]] = p['joint_paths'][0]
    b[p['additive_rows'][0]] = p['additive_paths'][0]
    i, j = np.triu_indices(len(cv), 1)
    assert bounds['additive']['mean_full_grid_relative_motion_envelope'] == pytest.approx(4./3.)
    for y in rng.normal(size=(30, *cv.shape)):
        gain = np.linalg.norm(a-y, axis=-1)-np.linalg.norm(b-y, axis=-1)
        ea, eb = a-y, b-y
        relative = np.linalg.norm(ea[i]-ea[j], axis=-1)-np.linalg.norm(eb[i]-eb[j], axis=-1)
        assert abs(gain.mean()) <= bounds['additive']['mean_full_grid_ade_envelope']+1e-12
        assert abs(gain[:, -1].mean()) <= bounds['additive']['mean_full_grid_fde_envelope']+1e-12
        assert abs(relative.mean()) <= bounds['additive']['mean_full_grid_relative_motion_envelope']+1e-12
