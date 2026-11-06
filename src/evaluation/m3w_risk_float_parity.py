"""Numerical parity on training scales, never a new decision tolerance."""
import numpy as np

NORMALIZED_TOLERANCE = 64 * np.finfo(np.float32).eps


def compare(after, before, training_scale):
    a,b,s = map(np.asarray,(after,before,training_scale))
    if (a.ndim!=2 or a.shape[1]!=2 or b.shape!=a.shape or s.shape!=(2,)
            or not all(np.isfinite(v).all() for v in (a,b,s)) or np.any(s<=0)):
        raise ValueError('Finite aligned risk scores and saved training-only scales required')
    np.testing.assert_allclose(a/s,b/s,rtol=NORMALIZED_TOLERANCE,atol=NORMALIZED_TOLERANCE)
    # This is a float-parity check, not permission to move the zero threshold.
    np.testing.assert_array_equal(a<=0,b<=0)
    return dict(bitwise_equal=bool(np.array_equal(a,b)),
        maximum_absolute_difference=float(np.max(np.abs(a-b))) if len(a) else 0.,
        maximum_normalized_difference=float(np.max(np.abs((a-b)/s))) if len(a) else 0.,
        normalized_tolerance=float(NORMALIZED_TOLERANCE),all_channel_decisions_exact=True,
        original_scalar_tolerance_violations=int((~np.isclose(a,b,rtol=1e-6,atol=1e-9)).sum()))
