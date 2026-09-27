import pytest
from scripts.report_m3w_european_temporal_support import contrasts, summarize


def rows():
    result = []
    def metrics(mse):
        return dict(status='measured', H_easy_MSE=mse, H_all_MSE=1,
            coverage=1., top10_mass_share=.3)
    for pair in ('full', 'motion_only'):
        for seed in (17, 29, 43):
            for inner in 'abcd':
                for outer in set('abcd')-{inner}:
                    result.append(dict(input=dict(pair=pair, producer=0, controller=1,
                        seed=seed, inner=inner, outer=outer), metrics=dict(new=metrics(.9), old=metrics(1.))))
    return result


def config(): return dict(seeds=[17, 29, 43], comparisons=[['new', 'old']], bootstrap_seed=7, bootstrap_draws=3000)


def test_contexts_and_seeds_not_independent_samples():
    c = contrasts(rows(), config()); v = c['full']['new_vs_old']['producer0_controller1']['primary']
    assert v['point'] == pytest.approx(10.) and v['localities'] == 4
    assert v['seeds'] == 3 and v['outer_contexts_per_site_seed'] == 3
    assert summarize(c)['full']['new_vs_old']['primary']['positive'] == 1


def test_missing_context_or_seed_cannot_be_silently_dropped():
    with pytest.raises(ValueError): contrasts(rows()[1:], config())


def test_missing_guard_support_retained():
    r = rows(); r[0]['metrics']['new']['coverage'] = 0
    out = contrasts(r, config())['full']['new_vs_old']['producer0_controller1']
    assert out['coverage_log']['status'] == 'not_estimable'
    assert out['primary']['status'] == 'measured'
