import numpy as np
import pytest

from scripts import run_m3w_cv_screen_order as study


def fixture():
    gain = np.array([[3.,2.,1.], [3.,2.,1.], [2.,2.,1.], [3.,2.,1.]])
    risk = np.array([[[1.,0.],[0.,0.],[0.,0.]], [[1.,0.],[0.,1.],[0.,0.]],
                     [[0.,0.],[0.,0.],[0.,0.]], [[0.,0.],[0.,0.],[0.,0.]]])
    legal = np.ones_like(gain,dtype=bool)
    legal[-1] = False
    return gain, risk, np.ones_like(gain), legal


def test_screen_before_nomination_preserves_only_admissible_candidates():
    pred = fixture()
    gate = (0.,.1)
    best = study.nominate(pred)
    np.testing.assert_array_equal(best[0],[0,0,0,-1])
    np.testing.assert_array_equal(study.nominate(pred,gate)[0],[1,2,0,-1])
    keep = np.array([[True,False,True]] * 4)
    np.testing.assert_array_equal(study.nominate(pred,gate,keep)[0],[2,2,0,-1])
    groups = [np.arange(4)]
    np.testing.assert_array_equal(study.base.choices(best,groups,gate),[-1,-1,0,-1])


def test_matching_preserves_zero_edit_queries_and_fails_for_missing_support():
    nom = study.nominate(fixture(),(0.,.1))
    groups = [np.array([0,1]),np.array([2,3])]
    control = np.array([0,-1,-1,-1])
    np.testing.assert_array_equal(study.choose(nom,groups,control),[1,-1,-1,-1])
    with pytest.raises(ValueError,match='insufficient'):
        study.choose(nom,groups,np.array([0,0,0,0]))
