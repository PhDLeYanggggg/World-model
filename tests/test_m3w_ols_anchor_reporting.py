import numpy as np

from scripts.report_m3w_ols_anchor_denoising import interval


def test_undefined_source_is_reported_not_dropped_or_filled():
    x = [np.array([1.,np.nan]),np.array([2.,np.nan])]
    got = interval(x,np.array([[0,0],[1,1]]))
    assert got == dict(point=None,interval95=None,missing_source_seed_cells=2)


def test_complete_slice_still_uses_equal_source_averaging():
    x = [np.array([1.,3.]),np.array([2.,4.])]
    got = interval(x,np.array([[0,0],[1,1]]))
    assert got['point'] == 2.5 and got['missing_source_seed_cells'] == 0
