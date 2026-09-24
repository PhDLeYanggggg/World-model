import numpy as np
import pytest
from src.data_unification.m3w_quantized_prefix import QuantizedPrefixAdapter


@pytest.mark.parametrize("factor", [.01, 1., 100.])
def test_identical_canonical_geometry_under_units(factor):
    p = np.array([[t,a,.002*t*(a+1),.03*a] for a in range(3) for t in range(8)])
    a = QuantizedPrefixAdapter(p,query_frame=7,recording_id="synthetic")
    changed = p.copy(); changed[:,2:] = changed[:,2:]*factor + [13.,-9.]
    b = QuantizedPrefixAdapter(changed,query_frame=7,recording_id="synthetic")
    np.testing.assert_array_equal(a.geometry_batch()[0],b.geometry_batch()[0])
    assert not b.metadata["lossless"]


def test_future_rejected_and_subprecision_motion_can_disappear():
    p = np.array([[t,a,float(a)+t*1e-12,0.] for a in range(2) for t in range(8)])
    r = QuantizedPrefixAdapter(p,query_frame=7,recording_id="synthetic")
    g,_ = r.geometry_batch()
    assert not g[:,:16].any()
    with pytest.raises(ValueError,match="Future"):
        QuantizedPrefixAdapter(np.r_[p,[[8,0,5.,6.]]],query_frame=7,recording_id="synthetic")
    with pytest.raises(PermissionError): r.get_labels(0)
