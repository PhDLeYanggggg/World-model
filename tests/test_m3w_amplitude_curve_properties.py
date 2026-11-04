import numpy as np
from tests.test_m3w_amplitude_cost import packet
from src.world_model import m3w_amplitude_cost as api


def test_true_positive_harm_curves_and_unknown_bounds_have_physical_amplitude_shape():
    inputs,target,valid,_=packet()
    p=api.causal_bank(**inputs)
    truth=api.targets(p,target,valid,1.)['moments']
    alpha=np.array(api.FRACTIONS)
    for component in (1,4):
        harm=truth[:,1:,component]
        assert np.all(np.diff(harm,axis=1)>=-1e-10)
        assert np.all(np.diff(harm/alpha,axis=1)>=-1e-10)
        assert np.all(harm<=p['causal']['envelope'][:,1:]+1e-10)


def test_reference_denominators_do_not_shrink_with_intervention_amplitude():
    inputs,target,valid,_=packet()
    p=api.causal_bank(**inputs)
    truth=api.targets(p,target,valid,1.)['moments']
    for component in (2,3):
        np.testing.assert_array_equal(truth[:,:,component],np.broadcast_to(truth[:,0,component,None],(4,7)))
    assert truth[3,1,3] == truth[3,6,3] > 0
