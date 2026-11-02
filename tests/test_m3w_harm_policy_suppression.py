import numpy as np
import pytest
from scripts.report_m3w_harm_policy_suppression import classify


def test_gain_and_guard_causes_partition_without_labels_or_safe_zero():
    p = np.tile([1., .001, 1., 1., .001], (5,7,1))
    p[0,:,0] = 0; p[1,:,1] = .1; p[2,:,0] = 0; p[2,:,1] = .1
    a = np.array([0,0,0,0,-1]); b = np.array([-1,-1,-1,0,-1])
    out = classify(a,b,p)
    assert [out[k] for k in ('gain_only','guard_only','both_gain_and_guard','neither_rejected')] == [1,1,1,1]
    assert out['old_selected'] == 4 and out['now_fallback'] == 3 and out['new_selected'] == 1
    with pytest.raises(ValueError): classify(a,b,np.full_like(p,np.nan))
