import numpy as np

from src.world_model.m3w_transfer_pass_transition import transition


def test_risk_qualified_state_only_changes_for_a_gain_or_loss():
    np.testing.assert_array_equal(transition([False,False,True,True],[False,True,False,True]),[0,1,2,0])
