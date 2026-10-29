from scripts.verify_m3w_query_conditioning import comparison


def probe(before, clipped, unclipped):
    return dict(fixed_self_score_MSE_before={'easy': before},
                virtual_steps={k: {'fixed_self_score_MSE': {'easy': v}}
                               for k, v in [('as_trained', clipped), ('unclipped', unclipped)]})


def test_missing_cohort_and_tolerance_are_not_improvement():
    result = comparison([probe(None, None, None), probe(1., .9, .9),
                         probe(1., .9, .8), probe(1., .9, 1.1)], 'easy', fixed=True)
    assert result['unclipped_minus_clipped']['n'] == 3
    assert result['better'] == result['worse'] == result['tied'] == 1
