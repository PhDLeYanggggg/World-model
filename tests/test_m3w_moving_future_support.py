import numpy as np
import pytest

from scripts.audit_m3w_moving_future_support import MISSING, UNKNOWN, audit, keys_and_categories, parse_csv


def annotation(agent, frames):
    return "".join(f"{t},{agent},{2*t},0,2,4,-1,-1,-1,-1\n" for t in frames)


def test_independent_query_and_missingness_reasons():
    rows = annotation(5, [*range(8), 23]) + annotation(7, range(9)) + annotation(9, [18])
    boxes, frames = parse_csv(rows.encode())
    result = {(q, i): (labels, kind) for q, i, labels, kind in keys_and_categories(boxes, frames, 25)}
    assert result[7, 5][1] == UNKNOWN[1]
    assert result[8, 7][1] == UNKNOWN[0]
    assert result[7, 5][0][0] == MISSING[2]
    assert result[7, 5][0][1] == MISSING[3]
    assert result[8, 7][0][0] == MISSING[1]
    with pytest.raises(ValueError, match="duplicate"):
        parse_csv((rows + annotation(5, [23])).encode())


def test_exact_recovery_is_raw_same_frame_not_interpolation():
    boxes, frames = parse_csv(annotation(5, range(9)).encode())
    target = np.zeros((2, 12, 2))
    valid = np.zeros((2, 12), bool)
    hist = np.zeros((2, 8, 2))
    hist[:, :, 0] = 2*np.arange(-7, 1)
    paths = np.zeros_like(target)
    result = audit(boxes, frames, 21, target, valid, hist, paths, paths, np.ones(2))
    assert result["exact_recoverable_steps"] == 1
    assert result["past_geometry_row_mismatches"] == 0
    assert result["unique_missing_track_frame_keys"] == 12
