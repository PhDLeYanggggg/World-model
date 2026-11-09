from itertools import product
from math import floor, ceil

import pytest

from reproducibility.reviewer_evidence.verify import decision, exact_site_interval
from scripts import package_m3w_reviewer_evidence as package


@pytest.mark.parametrize('values', [[1., 4.], [-3., 1., 1.], [4., 4., 4.], [0., -1., 5., 8.]])
def test_weighted_enumeration_equals_ordered_bootstrap(values):
    n = len(values)
    brute = sorted(sum(row)/n for row in product(values, repeat=n))
    expected = []
    for q in (.025, .975):
        idx = (len(brute)-1)*q
        low, high = brute[floor(idx)], brute[ceil(idx)]
        expected.append(low+(high-low)*(idx-floor(idx)))
    assert exact_site_interval(values) == pytest.approx(expected)


def test_decision_rejects_wrong_bound_orientation():
    data = dict(locations=7, recordings=493, views=8874, seeds=[17, 29, 43],
                contrasts={'test': dict(observed=[0.]*7, lower=[1.]*7)})
    with pytest.raises(ValueError, match='invalid paired'):
        decision(data)


def test_inherited_archive_must_match_the_bound_source(tmp_path):
    (tmp_path/'main_claims_replay.zip').write_bytes(b'altered')
    with pytest.raises(ValueError, match='Archived replay differs'):
        package.inherited('headline', tmp_path)
