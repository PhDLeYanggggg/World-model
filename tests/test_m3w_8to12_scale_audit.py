import numpy as np
import pytest

from scripts.audit_m3w_8to12_training_scale import energy_summary


def test_rare_large_targets_dominate_energy_but_are_not_deleted():
    rows = np.ones((100, 3))
    rows[-1] = [.001, 10000., 0.]
    result = energy_summary(rows)
    assert result['rows'] == 100
    assert result['top_approximately_one_percent_rows'] == 1
    assert result['top_energy_share'] == pytest.approx(10000 / 10099)
    assert result['scale_floor_rows'] == result['zero_history_path_rows'] == 1


def test_zero_energy_is_not_reported_as_a_ratio():
    rows = np.zeros((3, 3))
    rows[:, 0] = .001
    assert energy_summary(rows)['top_energy_share'] is None


@pytest.mark.parametrize('rows', [[], [[1, np.nan, 2]], [[1, -1, 2]], [[1, 2]]])
def test_invalid_inputs_are_rejected(rows):
    with pytest.raises(ValueError):
        energy_summary(rows)
