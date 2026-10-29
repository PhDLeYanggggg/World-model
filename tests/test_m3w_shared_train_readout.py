import numpy as np
import pytest

from scripts import evaluate_m3w_shared_train_risk as api
from tests.test_m3w_shared_train_risk import packet


def test_identical_actions_cancel_even_unknown_completion():
    args=packet();p=np.tile([.1,0.,1.,1.,0.],(32,1))
    result,_=api.evaluate(args,{k:p.copy() for k in api.ARMS})
    assert result['unknown_rows']==1 and not result['inference_label_filter']
    for key,v in result['contrasts'].items():
        assert abs(v)<1e-12


def test_empty_matched_risk_is_not_imputed_safe():
    args=packet();p=np.tile([.1,0.,1.,1.,0.],(32,1))
    preds={k:p.copy() for k in api.ARMS};preds['local'][:,0]=0
    result,_=api.evaluate(args,preds)
    policy=result['matched_policies']['shared_source_minus_local']['shared_source']
    assert policy['selected_count']==0 and policy['easy_selected_risk_upper'] is None
    assert not policy['finite_completion_supported']


def test_missing_arm_rejected():
    with pytest.raises(ValueError):api.evaluate(packet(),{})
