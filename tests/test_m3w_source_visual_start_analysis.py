import numpy as np
import pytest
from scripts.analyze_m3w_source_visual_start import paired_brier_parts


def test_variable_reference_brier_decomposition_is_exact():
    rng=np.random.default_rng(17)
    y=rng.integers(0,2,50); p=rng.random(50); ref=rng.random(50)
    d=paired_brier_parts(y,p,ref)
    assert np.isclose(d['total_lift'],d['mean_shift_contribution']+d['within_site_variation_contribution'])
    assert d['held_label_diagnostic_not_inference_calibration']


def test_constant_prior_repair_is_not_within_site_information():
    y=np.array([0.,1.,0.,1.]); p=np.full(4,.5); ref=np.full(4,.8)
    d=paired_brier_parts(y,p,ref)
    assert np.isclose(d['mean_shift_contribution'],.09) and d['within_site_variation_contribution']==0
    assert paired_brier_parts(y,y,p)['within_site_variation_contribution']==.25
    with pytest.raises(ValueError):
        paired_brier_parts(y,p[:2],ref)
