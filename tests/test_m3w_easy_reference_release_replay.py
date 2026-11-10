from copy import deepcopy
from itertools import product

import pytest

from reproducibility.easy_reference_release_supplement import verify


def panel():
    directions = []
    for h, c, s in product(range(7), range(7), range(3)):
        if h == c:
            continue
        empty = dict.fromkeys((*verify.COUNTS, 'calibration_count', 'lower_cv_pp'), 0)
        empty['eligible'] = 35
        directions.append(dict(roles=[h, c, s], arms={a: empty.copy() for a in verify.ARMS}))
    directions[0]['arms']['sign'].update(released=32, risk=12, dual=8, strict=6, missing=16,
                                        calibration_count=1, lower_cv_pp=.18)
    return dict(directions=directions, draws=[list(range(7))]*3000)


def test_replay_uses_paired_location_means():
    result = verify.replay(panel())
    assert result['counts']['sign']['strict'] == 6
    assert result['counts']['reference_only']['risk'] == 0
    assert result['paired']['reference_only']['point_pp'] == pytest.approx(-.18/126)


def test_replay_rejects_new_nominations_and_unreleased_threshold():
    p = panel()
    different = deepcopy(p)
    different['directions'][1]['arms']['reference_only']['eligible'] += 1
    with pytest.raises(ValueError, match='different nomination rosters'):
        verify.replay(different)
    p['directions'][1]['arms']['reference_only']['released'] += 1
    with pytest.raises(ValueError, match='Inconsistent release receipt'):
        verify.replay(p)
