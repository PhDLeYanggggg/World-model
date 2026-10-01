import numpy as np
import pytest
from src.world_model.m3w_source_policy_selection import choose, validation_readout


def test_utility_choice_and_fixed_tie_order():
    y=np.array([[1.,0.,2.,2.,0.],[.5,0.,2.,2.,0.]])
    actions={'mse':np.array([True,False]),'final':np.array([True,True]),'initial':np.array([False,False])}
    assert choose(y,actions)['selected']=='final'
    actions['mse']=actions['final'].copy()
    assert choose(y,actions)['selected']=='mse'


def test_net_benefit_cannot_cancel_positive_harm_budget():
    y=np.array([[2.,0.,2.,2.,0.],[0.,.2,2.,2.,.2]])
    z=validation_readout(y,np.ones(2,bool))
    assert z['all_gain_fraction']>0 and not z['eligible']
    assert 'all_positive_harm_exceeds_budget' in z['reasons']


def test_unknown_labels_do_not_become_harmless_or_modify_row_actions():
    y=np.array([[1.,0.,2.,2.,0.],[np.nan]*5]); take=np.ones(2,bool)
    z=validation_readout(y,take)
    assert z['selected_unknown']==1 and not z['eligible']
    np.testing.assert_array_equal(take,[True,True])


def test_empty_and_missing_easy_reference_are_not_safety_passes():
    y=np.array([[1.,0.,2.,0.,0.]])
    assert not validation_readout(y,np.array([True]))['eligible']
    actions={k:np.zeros(1,bool) for k in ('mse','final','initial')}
    z=choose(y,actions)
    assert z['selected']=='fallback' and z['status']=='no_supported_nonempty_policy'
    assert not z['population_safety_guarantee']


def test_shapes_and_partial_unknown_rejected():
    with pytest.raises(ValueError):validation_readout(np.zeros((2,4)),np.ones(2,bool))
    with pytest.raises(ValueError):validation_readout(np.array([[np.nan,0,1,1,0]]),np.ones(1,bool))
    with pytest.raises(ValueError):choose(np.zeros((1,5)),{})
