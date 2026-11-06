import numpy as np

from scripts.run_m3w_moving_donor_seeds import summarize


def example():
    records, contexts = [], []
    for j, sites in enumerate((['a', 'b'], ['c', 'd', 'e'], ['f', 'g', 'h'])):
        job = str(j)
        forest = dict(total_joint_cost=float(np.mean(np.arange(1, len(sites)+1))), false_safe_cost=0.)
        contexts.append(dict(job=job, source_macro={'shared64': forest}))
        for seed in (17, 29, 43):
            for k, site in enumerate(sites, 1):
                for arm, factor in [('own_only', 1.), ('A_only', .9), ('augmented', .8)]:
                    metrics = dict(total_joint_cost=k*factor, false_safe_cost=.1*factor,
                                   false_reject_cost=k*factor-.1*factor, normalized_MSE=2*factor)
                    records.append(dict(job=job, seed=seed, source=site, arm=arm, metrics=metrics,
                                        forest_metrics={'total_joint_cost': float(k)}))
    return records, dict(contexts=contexts)


def test_known_paired_effect_uses_all_sources_and_seeds():
    records, frozen = example()
    result = summarize(records, frozen)
    assert result['replication_screen_pass']
    assert result['passing_contexts_by_seed'] == {'17': 3, '29': 3, '43': 3}
    np.testing.assert_allclose(result['source_bootstrap_vs_own']['interval95'], [20, 20])
    assert not result['deployment_changed']
    assert result['forecaster_seeds'] == [17]


def test_one_failed_seed_is_not_hidden_by_positive_average():
    records, frozen = example()
    for r in records:
        if r['seed']==43 and r['arm']=='augmented':
            r['metrics']['total_joint_cost'] *= 1.5
    result = summarize(records, frozen)
    assert not result['replication_screen_pass']
    assert result['passing_contexts_by_seed']['43'] == 0
