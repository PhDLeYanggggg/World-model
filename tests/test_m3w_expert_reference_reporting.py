from scripts.report_m3w_moving_expert_reference import candidate_screen,utility_supported


def test_positive_utility_is_not_safety_and_missing_primary_is_not_pass():
    a=dict(primary=dict(completion_lower_pp=dict(interval95=[.1,.3])),
        policies=dict(expert_context=dict(joint_free=dict(safety=dict(required_record_views=72,original_record_contract_pass=False)))))
    assert utility_supported(a) and not candidate_screen(a)
    a['policies']['expert_context']['joint_free']['safety']['original_record_contract_pass']=True
    assert candidate_screen(a)
    a['primary']['completion_lower_pp']['interval95']=None
    assert not utility_supported(a) and not candidate_screen(a)
