import numpy as np
import pytest

from scripts.audit_m3w_moving_feature_support import proposal_features, reference, describe
from tests.test_m3w_admission_priority import prepared


def test_fallback_encoding_and_no_future_fields(tmp_path):
    d, warm, _ = prepared(tmp_path)
    choice = np.array([-1, 1, 2, 3])
    z = proposal_features(d['causal'], warm, choice)
    assert z.shape == (4, 444) and np.isfinite(z).all()
    np.testing.assert_array_equal(z[0], proposal_features(d['causal'], warm, np.maximum(choice, 0))[0])
    active = z[choice >= 0]
    ref = reference(active); r = describe(active, ref)
    assert r['rows'] == 3 and r['rows_any_outside_fitting_range_fraction'] == 0
    with pytest.raises(ValueError, match='causal schema'):
        proposal_features(dict(d['causal'], future=np.zeros((4, 2))), warm, choice)


def test_fitting_reference_is_not_updated_by_held_inputs():
    a = np.zeros((3, 444)); ref = reference(a)
    b = np.ones((2, 444))*9
    r = describe(b, ref)
    assert r['rows_any_clipped_fraction'] == 1
    assert r['rows_any_outside_fitting_range_fraction'] == 1
    np.testing.assert_array_equal(ref['max'], np.zeros(444))
