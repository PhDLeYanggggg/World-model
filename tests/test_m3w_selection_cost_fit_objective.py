import numpy as np
import pytest
from src.evaluation.m3w_selection_cost_fit_objective import score


def test_complete_query_weighting_and_wholly_unlabeled_query_stays_zero_not_a_label():
    p = np.ones((3,7,5)); truth = np.zeros_like(p); truth[2] = np.nan
    allowed = np.ones((3,7),bool); known = np.array([True,True,False])
    out = score(p,truth,allowed,known,np.ones(3),np.ones(5),[np.array([0,1]),np.array([2])],
                np.array([4,6]),np.array(['a','a','a']))
    assert out['total'] == pytest.approx(.4)
    assert out['by_site']['a']['components']['EH'] == pytest.approx(.4)
    assert out['by_site']['a']['unlabeled_queries'] == 1
    assert out['unknown_supervised_rows'] == 0


def test_exact_zero_decoded_error_and_causal_scale():
    truth = np.arange(70).reshape(2,7,5)/100
    p = truth*np.array([2.,3.])[:,None,None]
    out = score(p,truth,np.ones((2,7),bool),np.ones(2,bool),np.array([2.,3.]),np.ones(5),
                [np.array([0]),np.array([1])],np.array([4,5]),np.array(['a','b']))
    assert out['total'] == pytest.approx(0.,abs=1e-30)
    with pytest.raises(ValueError):
        score(p,truth,np.ones((2,7),bool),np.ones(2,bool),np.zeros(2),np.ones(5),
              [np.array([0]),np.array([1])],np.array([4,5]),np.array(['a','b']))
