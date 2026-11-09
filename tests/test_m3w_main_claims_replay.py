import pytest

from reproducibility.main_claims.verify import summarize


def example():
    return {
        'opendd': [{'hindsight_status': {'verified_positive_both': 1, 'unresolved': 1, 'zero_quota': 1},
                    'past_only': [[17, 1, 0], [29, 1, 1], [43, 1, 0]]}],
        'moving': [[[17, 0.], [29, -1.], [43, 1.]]],
        'wepdtof': [[[[17, True, 1., 1., 1., True], [29, False, 1., 1., 1., False],
                      [43, False, 0., None, None, False]]]],
    }


def test_view_counts_keep_unresolved_distinct_and_require_both_references():
    out = summarize(example())
    assert out['opendd'] == dict(locations=1, recordings=1, views=3, hindsight_passes=1,
                                 past_only_passes=1, unresolved=1)
    assert out['moving']['ols4_excluded'] == 2
    assert out['wepdtof']['fixed_joint_dual_passes'] == 1


def test_reject_inconsistent_external_pass_and_duplicate_seeds():
    bad = example()
    bad['wepdtof'][0][0][0][4] = 0.
    with pytest.raises(ValueError, match='dual-pass'):
        summarize(bad)
    bad = example()
    bad['moving'][0][0][0] = 29
    with pytest.raises(ValueError, match='frozen seed'):
        summarize(bad)
