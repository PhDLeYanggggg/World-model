import numpy as np
import pytest
from scripts.run_m3w_european_eqmotion_controls import aligned_rows


def test_control_population_join_is_exact_and_never_silently_drops_rows():
    ids=np.array([1,3,7,9])
    np.testing.assert_array_equal(aligned_rows(ids,np.array([3,9])),[1,3])
    for bad in (np.array([3,8]),np.array([3,3]),np.array([9,3]),np.array([3.,9.])):
        with pytest.raises(ValueError):aligned_rows(ids,bad)
    with pytest.raises(ValueError):aligned_rows(np.array([1,1]),np.array([1]))
