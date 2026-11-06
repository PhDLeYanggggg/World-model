from scripts.report_m3w_moving_complete_risk_donors import eligible


def test_partial_support_never_becomes_pass_from_positive_ade():
    p=dict(safety=dict(required_record_views=24,original_record_contract_pass=False),
           source_macro_completion_lower_gain=99.)
    assert not eligible(p)
    p['safety']['original_record_contract_pass']=True
    assert eligible(p)
    p['source_macro_completion_lower_gain']=0
    assert not eligible(p)
    p['safety']['required_record_views']=1
    p['source_macro_completion_lower_gain']=1
    assert not eligible(p)
