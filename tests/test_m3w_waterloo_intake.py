import numpy as np

from src.evaluation import m3w_waterloo_intake as m


def test_lattice_accepts_quantization_but_not_gaps_as_new_cadence():
    ts = np.round(np.arange(15)*1001/30000, 6)
    step, jitter = m.ticks([ts[:-4], ts[3:]])
    assert step == 1001/30000 and jitter < 1e-6
    assert m.ticks([np.array([0., .05321, .09443])]) == (None, None)


def test_history_only_and_sparse_future_counts():
    a = np.arange(24)
    r, past, full = m.support(a, 1)
    assert len(past) == 17 and len(full) == 5
    assert r["no_future12"] == 1 and r["partial_future12"] == 11
    assert r["raw_endpoint_10"] == 7
    _, before, _ = m.support(a[a <= 13], 1)
    np.testing.assert_array_equal(before, past[past <= 13])
    b = np.delete(a, 9)
    _, h, _ = m.support(b, 1)
    assert 10 not in h and 17 in h


def test_strided_counts_cover_all_phases_without_interpolation():
    r, h, f = m.support(np.arange(240), 12)
    assert len(h) == 240-7*12 and len(f) == 240-19*12
    assert r["past_eligible"] == r["complete_future12"]+r["partial_future12"]+r["no_future12"]
