import numpy as np
import pytest

from scripts.analyze_m3w_8to12_error_concentration import concentration


def test_small_scale_can_dominate_normalized_but_not_native_error():
    rows = [[.001, 1000., 999.], *[[1., 1., 1.1]] * 99]
    out = concentration(rows)
    tail = out['top_approximately_one_percent_by_normalized_floor_error']
    assert tail['rows'] == 1
    assert tail['floor_normalized_error_share'] == pytest.approx(1000 / 1099)
    assert tail['floor_native_error_share'] == .01
    assert out['scale_at_most_0p01_dataset_local']['rows'] == 1


def test_zero_error_is_not_a_fraction_or_improvement():
    out = concentration([[1, 0, 0]])
    assert out['all']['floor_normalized_error_share'] is None
    assert out['all']['floor_native_error_share'] is None


@pytest.mark.parametrize('rows', [[], [[0, 1, 2]], [[1, np.nan, 0]], [[1, -1, 0]]])
def test_invalid_diagnostic_rows_rejected(rows):
    with pytest.raises(ValueError):
        concentration(rows)
