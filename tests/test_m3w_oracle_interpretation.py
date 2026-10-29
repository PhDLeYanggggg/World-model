import numpy as np
from src.evaluation.m3w_neural_marginal_ceiling import candidate_ceiling


def test_zero_physical_oracle_gap_does_not_prove_neural_causal_uselessness():
    # Three indistinguishable futures at equilateral-triangle vertices.
    # A fixed center forecast beats each fixed vertex on mean Euclidean loss,
    # even though a future-informed vertex oracle already has zero loss.
    edge=np.sqrt(3.)
    physical=np.array([[0.,edge,edge],[edge,0.,edge],[edge,edge,0.]])
    ade=np.c_[physical,physical,np.ones(3)]
    value=candidate_ceiling(ade,physical[:,0],np.ones((3,7),bool),
        np.ones(3,bool),np.ones(3,bool),np.ones(3))
    assert value['all']['neural_marginal_gain_floor_pp']==0.
    assert ade[:,6].mean()<physical.mean(0).min()
