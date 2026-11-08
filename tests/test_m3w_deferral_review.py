from copy import deepcopy

import pytest

from scripts.build_m3w_deferral_review import data
from src.reproduction.m3w_deferral_review import match, replay


def test_latest_numerical_bundle_replays_and_detects_mutation():
    stats, expected, _ = data()
    cells = stats["study_4"]
    match(replay(cells), expected["study_4"])
    wrong = deepcopy(cells)
    wrong[0][0]["direct"]["priced_vs_direct"] += 1.
    with pytest.raises(AssertionError):
        match(replay(wrong), expected["study_4"])
    with pytest.raises(ValueError, match="three paired seed"):
        replay([*cells[:-1], cells[-1][:2]])
