import numpy as np
from scripts.report_m3w_european_support_factorization import POLICIES, REASONS, SUBSETS, partition_summary, EXPECTED_COUNTS


def test_complete_registered_policy_and_replay_counts():
    assert len(POLICIES) == len(set(POLICIES)) == 13
    assert EXPECTED_COUNTS['saved_decisions_verified'] == 36*2*13
    assert EXPECTED_COUNTS['old_metrics_exact'] == 36*2*4*6
    assert EXPECTED_COUNTS['independent_metric_reductions'] == 36*348


def test_partition_summary_equal_locality_not_pooled():
    entries = {}
    for site, denominator, harm, benefit in [('a', 20, 2, 1), ('b', 100, 1, 3)]:
        cats = {n: dict(indexed_rows=3, known_rows=2, harm_pp=100*harm/denominator,
                       benefit_pp=100*benefit/denominator) for n in REASONS}
        entries[site] = dict(floor_error_sum=denominator, categories=cats)
    out = partition_summary([{s: entries for s in SUBSETS}])['all']['history_only_failure']
    np.testing.assert_allclose(out['avoided_harm_pp'], [5.5, 5.5])
    np.testing.assert_allclose(out['lost_benefit_pp'], [4, 4])
    np.testing.assert_allclose(out['removal_change_pp'], [1.5, 1.5])
    assert out['indexed_rows'] == [6, 6] and out['known_rows'] == [4, 4]


def test_undefined_partition_does_not_become_zero():
    entries = {'a': dict(floor_error_sum=0, categories={n: dict(indexed_rows=3, known_rows=0,
        harm_pp=None, benefit_pp=None) for n in REASONS})}
    out = partition_summary([{s: entries for s in SUBSETS}])['all']['source_overlap_failure']
    assert out['avoided_harm_pp'] is None and out['removal_change_pp'] is None
    assert out['indexed_rows'] == [3, 3] and out['known_rows'] == [0, 0]
