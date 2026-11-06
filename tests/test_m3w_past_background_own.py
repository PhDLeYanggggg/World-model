import numpy as np
from scripts.run_m3w_moving_past_background_own import past_members
from scripts.run_m3w_moving_past_background import PAST_KEYS


def test_only_past_npz_members_are_materialized(tmp_path):
    path = tmp_path/'members.npz'
    fields = {k: np.arange(3) for k in PAST_KEYS}
    # This member raises if accessed with allow_pickle=False,not when ignored.
    fields['target_eval'] = np.array([{'future': 'forbidden'}], dtype=object)
    np.savez(path, **fields)
    actual = past_members(path)
    assert set(actual) == set(PAST_KEYS) and 'target_eval' not in actual
    for key in PAST_KEYS:
        np.testing.assert_array_equal(actual[key], fields[key])
