import pytest

from scripts.check_m3w_dronecrowd_combined_coverage import covered_trials


def test_broad_roster_only_covers_exact_recording_ids():
    trials = [dict(key='fold2_seed17', lineage=dict(recordings=['a', 'b', 'c'], unscreened=['b', 'c']))]
    result = covered_trials(trials, {'a'}, {'a', 'b', 'C'})
    assert result == [dict(key='fold2_seed17', training_recordings=3,
                           added_coverage=['b'], still_uncovered=['c'])]


def test_join_requires_the_original_training_lineage():
    trials = [dict(key='fold2_seed17', lineage=dict(recordings=['a', 'b'], unscreened=[]))]
    with pytest.raises(ValueError, match='Training roster'):
        covered_trials(trials, {'a'}, {'b'})
