import numpy as np
import pytest

from src.evaluation.m3w_moving_forecast_support import fitting_design_with_empty_groups,qualify_supported_analysis
from src.world_model.m3w_european_source_forecast import fit_design


def test_empty_requested_group_does_not_drop_any_rows_or_allow_unknown_group():
    data=dict(sites=np.array(['a','b','a','b']))
    errors=np.ones((4,6))
    result=fitting_design_with_empty_groups(fit_design,data,['a','empty'],errors,['empty'])
    np.testing.assert_array_equal(result['train_ids'],[0,2]);np.testing.assert_array_equal(result['held_ids'],[1,3])
    assert result['declared_but_structurally_empty_groups']==['empty']
    with pytest.raises(ValueError,match='Unexpected'):
        fitting_design_with_empty_groups(fit_design,data,['a','typo'],errors,['empty'])


def test_supported_positive_result_cannot_pass_missing_full_group_primary():
    raw=dict(primary=dict(point=5.,CI95=[1.,7.]),known=dict(point=6.),scope='old')
    result=qualify_supported_analysis(raw,['a','b','empty'],['empty'])
    assert result['primary']['point'] is None and result['primary']['status']=='not_estimable'
    assert result['supported_group_completion_lower_secondary']['point']==5.
    assert not result['all_declared_group_primary_passed']
    assert raw['primary']['point']==5. and 'known' in raw
