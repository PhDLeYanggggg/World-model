import numpy as np
import pytest

from scripts.run_m3w_okutama_label_histories import forecasts, parse, prefixes, trajectories, windows


def test_nongenerated_scope_excludes_generated_and_lost_rows():
    raw = (b'1 0 0 2 2 0 0 0 0 "Person"\n'
           b'1 1 1 3 3 1 0 0 1 "Person"\n'
           b'1 1 1 3 3 2 1 0 0 "Person"\n')
    a = parse(raw, [0, 1, 2])
    x = trajectories(a, 3, "nongenerated")[:, 0]
    np.testing.assert_array_equal(x[0], [1., 1.])
    assert np.isnan(x[1:]).all()
    np.testing.assert_array_equal(trajectories(a, 3, "released")[1, 0], [2., 2.])
    with pytest.raises(ValueError, match="Duplicate"):
        parse(raw + raw.splitlines()[0] + b"\n", [0, 1, 2])


def test_future_mutation_does_not_change_prefix_or_forecast():
    h = np.broadcast_to(np.arange(20.)[:, None, None], (20, 2, 2)).copy()
    h[3, 1] = np.nan
    alt = h.copy()
    alt[8:] = np.nan
    a, b = prefixes(h, 1), prefixes(alt, 1)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a[1], [0])
    np.testing.assert_array_equal(forecasts(a[2])["hindcast"], forecasts(b[2])["hindcast"])
    np.testing.assert_array_equal(windows(160, 4), [0, 80])
