import pytest
from src.evaluation import m3w_record_balanced_readout as api


def test_empty_match_risk_not_safe_and_new_arm_roster():
    from tests.test_m3w_direct_policy import fixture as packets_fixture
    packets,pred=packets_fixture();args=next(iter(packets.values()));n=len(args[0])
    masks={'origin':api.np.ones(n,bool),'cost_sensitive':api.np.zeros(n,bool),
           'record_balanced':api.np.arange(n)%2==0}
    ranks={a:api.np.arange(n,dtype=float) for a in api.ARMS}
    value=api.evaluate(args,masks,ranks)
    pair=value['pairs']['record_balanced_minus_cost_sensitive']
    assert pair['matched_empty'] and pair['matched_count']==0
    assert pair['matched_right']['easy_selected_risk_upper'] is None
    assert not api.budget_status(pair['matched_right'])['supported']
    for arm in api.ARMS:
        assert sum(x[arm]['known_rows'] for x in value['recordings'].values())==value['policies'][arm]['known_rows']
    with pytest.raises(ValueError):api.summarize([])
