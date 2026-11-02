import numpy as np
import pytest

from scripts.verify_m3w_selection_bank_support import recording_views


@pytest.mark.parametrize('records', [np.array([21,21,35]),np.array(['21','21','35'])])
def test_serialized_keys_keep_native_recording_comparison(records):
    views = list(recording_views(records, {'21': {'rows': 2},'35': {'rows': 1}}))
    assert [int(at.sum()) for at,_ in views] == [2,1]
    assert all(int(at.sum()) == row['rows'] for at,row in views)


def test_serialized_recording_roster_mismatch_is_not_silently_dropped():
    with pytest.raises(ValueError,match='rosters differ'):
        list(recording_views(np.array([21,35]), {'21': {'rows': 2}}))
