from scripts.report_m3w_moving_source_forecast import macro


def test_full_supported_group_endpoint_summary_does_not_drop_missing_group():
    rows=[]
    for group in ('a','b'):
        for k in range(6):
            rows.append(dict(site=group,scores=dict(eqmotion=dict(available_ADE=2.,endpoint_FDE=None if group=='b' else 3.))))
    assert macro(rows,'eqmotion','available_ADE')==2.
    assert macro(rows,'eqmotion','endpoint_FDE') is None


def test_macro_requires_two_producers_and_three_seeds_per_group_not_partial_runs():
    rows=[dict(site='a',scores=dict(PCT=dict(available_ADE=2.))) for _ in range(5)]
    assert macro(rows,'PCT','available_ADE') is None
