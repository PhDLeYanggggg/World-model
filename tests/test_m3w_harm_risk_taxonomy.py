import copy
import pytest
from scripts.report_m3w_harm_risk_taxonomy import category


@pytest.mark.parametrize('change,expected',[
    ({},'supported'),
    ({'selected':0,'supported':False,'known_easy_risk':None},'no_path_change'),
    ({'known_easy_risk':None,'supported':False},'active_support_undefined'),
    ({'known_all_risk':.03,'supported':False},'known_contract_violation'),
    ({'completion_easy_risk_upper':.03,'supported':False},'unknown_completion_not_certified')])
def test_priority_partition_never_changes_the_original_gate(change,expected):
    record=dict(selected=1,known_all_risk=.01,known_easy_risk=.01,
        known_easy_net_degradation=.005,completion_all_risk_upper=.01,
        completion_easy_risk_upper=.01,supported=True)
    record.update(change);before=copy.deepcopy(record)
    assert category(record)==expected
    assert record==before


def test_inconsistent_unsupported_record_is_not_silently_passed():
    record=dict(selected=1,known_all_risk=.01,known_easy_risk=.01,
        known_easy_net_degradation=.005,completion_all_risk_upper=.01,
        completion_easy_risk_upper=.01,supported=False)
    with pytest.raises(ValueError):category(record)
