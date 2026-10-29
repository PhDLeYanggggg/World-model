from scripts.report_m3w_balanced_region_cost import render,NAME


def test_compute_and_conditional_statistics_do_not_become_safety_or_neural_success():
    contrasts={f'{mode}_{kind}_pp':dict(point=0.,CI95=[-1.,1.])
        for mode in ('full','matched') for kind in ('known','lower','upper')}
    value=dict(policies=[],exact_minus_stochastic={'inner_held':contrasts},
        inner_held_mixed_minus_physical={NAME:contrasts},auxiliary_teacher_rows=50,auxiliary_forward_rows=100,
        training={NAME:dict(fits=54,updates=108000,seconds=1.,peak_RSS_bytes=100)},
        full_roster_supported=False,neural_increment_supported=False)
    text=render(value)
    assert 'gate: False' in text and 'neural-increment CI: False' in text
    assert 'not safe zero' in text and 'not confirmation' in text
    assert 'do not mean identical training FLOPs' in text
    assert 'not three new full forecasting pipelines' in text
