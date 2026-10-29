import numpy as np
from src.evaluation.m3w_neural_marginal_ceiling import candidate_ceiling


def test_future_oracle_ceiling_keeps_eligibility_unknowns_and_fallback_distinct():
    ade=np.array([[8.,9,9,9,9,9,3],[2.,2,2,2,2,2,1],[np.nan]*7])
    floor=np.array([10.,4.,np.nan]);allowed=np.ones((3,7),bool);allowed[1,6]=False
    value=candidate_ceiling(ade,floor,allowed,np.array([True,True,False]),np.array([True,False,False]),np.ones(3))
    assert value['all']['physical_oracle_gain_floor_percent']==100*4/14
    assert value['all']['neural_marginal_gain_floor_pp']==100*5/14
    assert value['all']['neural_strictly_better_rows']==1
    assert value['unknown_rows']==1 and value['oracle_is_deployable'] is False
    assert value['labels_exported_as_inference_inputs'] is False


def test_zero_reference_is_not_an_oracle_improvement():
    value=candidate_ceiling(np.ones((2,7)),np.zeros(2),np.zeros((2,7),bool),
        np.ones(2,bool),np.zeros(2,bool),np.ones(2))
    assert value['all']['neural_marginal_gain_floor_pp'] is None
    assert value['hard']['known_rows']==0
