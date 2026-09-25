import pytest
from scripts.report_m3w_european_reference_protection import summarize,matched_fit_summary


def test_incomplete_reference_study_is_not_complete():
    with pytest.raises(ValueError): summarize([],{})
    with pytest.raises(ValueError): matched_fit_summary([])


def test_fit_comparison_is_matched_not_best_seed():
    rows=[]
    for pair in ('full','motion_only'):
        for _ in range(18):
            rows.append(dict(pair=pair,arms={a:dict(fit=dict(trace=[dict(moment_mse=v,
                component_mse=[v]*4)])) for a,v in (('continued',2.),('protected',1.))}))
    out=matched_fit_summary(rows)
    assert out['full']['protected_lower_count']==18
    assert out['motion_only']['fixed_batch_MSE_protected_over_continued_median']==.5
    assert out['full']['not_validation']
