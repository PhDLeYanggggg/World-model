from scripts import report_m3w_easy_harm_readout_v3 as report


def test_undefined_interval_is_not_zero_or_a_pass():
    assert report.interval(dict(mean=None,CI95=None)) == 'undefined; support retained'


def test_direction_and_precision_of_interval():
    assert report.interval(dict(mean=-.2,CI95=[-.3,-.1])) == '-0.200000 [-0.300000, -0.100000]'


def test_all_comparators_and_matched_safety_retained():
    contrasts = {}
    for other in report.run.original.api.COMPARATORS:
        for metric in ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'):
            contrasts[other+'_'+metric] = dict(mean=-1.,CI95=[-2.,0.])
    keys = ('selected_occurrences','unknown_selected_occurrences','undefined_easy_risk',
            'known_easy_violations','easy_upper_violations','finite_completion_supported')
    safety = {'easy_deviance':dict(zip(keys,[5,1,2,3,4,0])),
              'easy_deviance_matched_quadratic':dict(zip(keys,[0,0,72,0,0,0]))}
    a, b = report.tables(dict(contrasts=contrasts,safety=safety))
    for other in report.run.original.api.COMPARATORS:
        assert '| '+other+' |' in a
    assert 'easy_deviance_matched_quadratic' in b
    assert '| 0 | 0 | 72 | 0 | 0 | 0 |' in b
