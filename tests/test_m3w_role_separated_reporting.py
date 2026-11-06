import numpy as np
from scripts.report_m3w_moving_role_separated_cost import binary_quality,candidate_screen,mean_required


def test_one_class_label_support_is_na_not_false_auroc_success():
    q=binary_quality(np.zeros(3,bool),np.array([.1,.1,.1]),np.ones(3))
    assert q['AUROC'] is None and q['AUPRC'] is None
    assert np.isclose(q['Brier'],.01) and np.isclose(q['ECE'],.1)


def test_candidate_needs_all_views_and_registered_paired_support():
    p=dict(safety=dict(required_record_views=72,original_record_contract_pass=False))
    a=dict(policies=dict(role_separated=dict(joint_free=p)),primary=dict(completion_lower_pp=dict(interval95=[1,2])))
    assert not candidate_screen(a)
    p['safety']['original_record_contract_pass']=True;assert candidate_screen(a)
    a['primary']['completion_lower_pp']['interval95']=[-1,2];assert not candidate_screen(a)
    assert mean_required([dict(v=1),dict(v=None)],'v') is None
