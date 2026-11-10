from copy import deepcopy
import json

import numpy as np
import pytest

from reproducibility.m3w_reader_supplement_cross_grid.verify import check_wep
from scripts.audit_m3w_wepdtof_full_grid import OUT, gain_bounds


def test_gain_bounds_attained_without_changing_observed_mass():
    ref = np.zeros((3, 12, 2))
    p = ref + 1.
    y = ref + .8
    valid = np.ones((3, 12), bool)
    valid[0, -2:] = False
    valid[2, -1] = False
    w = np.array([.5, .5, 1.])
    mass = float(w @ np.where(valid, np.linalg.norm(ref-y, axis=-1), 0.).sum(1))
    _, low, high = gain_bounds(ref, p, y, valid, w, mass)
    for witness, edge in ((ref, low), (p, high)):
        z = np.where(valid[..., None], y, witness)
        assert gain_bounds(ref, p, z, np.ones_like(valid), w, mass) == pytest.approx((edge, edge, edge))
    assert gain_bounds(ref, p, y, valid, w, 0.) == (None, None, None)


def test_published_cross_grid_table_rejects_incompatible_joint_certificates():
    data = json.loads((OUT/'readout.json').read_text())
    assert check_wep(data)['fixed_joint']['full_pass_utility_excluded'] == 3
    bad = deepcopy(data)
    bad['arms']['fixed_joint']['full_grid_dual_guaranteed'] = 2
    with pytest.raises(ValueError, match='Inconsistent'):
        check_wep(bad)
