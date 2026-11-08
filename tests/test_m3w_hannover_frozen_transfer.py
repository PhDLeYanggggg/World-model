import numpy as np
import pytest

from scripts.audit_m3w_hannover_radius_capacity import capacity
from scripts.run_m3w_hannover_frozen_transfer import parse_csv, publisher_contract, windows


def test_archive_parser_binds_units_and_keeps_clip_local_tracks():
    assert publisher_contract("Each time step lasts 0.5 seconds\nRatio from pixel to meter\n21.18:1") == (21.18, .5)
    header = "time_step,object_id,x_coordinate,y_coordinate,user_type\n"
    two = "0,1,0.1,1.2,1\n1,2,3.5,4.6,1\n"
    grid, tracks, sigs = parse_csv((header+two).encode())
    assert grid.shape == (2, 2, 2) and set(tracks) == {1, 2} and len(sigs) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        parse_csv((header+two+"1,2,3.5,4.6,1\n").encode())


def test_future_missingness_does_not_change_native_query_or_history():
    data = np.arange(80, dtype=float).reshape(2, 20, 2)
    h, y, ids = windows(data, 2.)
    data[:, 8:] = np.nan
    a, b, rows = windows(data, 2.)
    np.testing.assert_array_equal(a, h)
    assert ids == rows == [0] and np.isfinite(y).all() and np.isnan(b).all()


def test_radius_ceiling_uses_only_possible_known_gain_but_risk_debits_unknown():
    cv = np.zeros((2, 2, 2))
    y = np.array([[[1., 0.], [3., 0.]], [[np.nan, np.nan], [np.nan, np.nan]]])
    valid = np.isfinite(y).all(2)
    a = capacity(cv, np.array([2., .5]), y, valid, 2.)
    assert a == {"optimistic_utility_upper_pp": 75., "unknown_only_easy_risk_lower_pct": 25., "easy_rows": 1}
