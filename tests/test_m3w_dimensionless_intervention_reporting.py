import numpy as np
from scripts.diagnose_m3w_dimensionless_intervention import fallback_reasons
from src.world_model.m3w_dimensionless_intervention import allowed


def test_diagnostic_replays_float64_threshold_boundary():
    u=np.array([[2.,1.]],np.float32)
    a=np.array([[7.,.14]],np.float32)
    e=np.array([[20.,.1]],np.float32)
    moving=np.ones(1,bool)
    assert bool(a[0,1] <= np.float32(.02)*a[0,0])
    actual=allowed(u,a,e,moving,.02)
    assert not actual[0]
    np.testing.assert_array_equal(actual,fallback_reasons(u,a,e,moving,.02)==4)
