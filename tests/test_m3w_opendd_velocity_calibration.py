import numpy as np

from src.evaluation.m3w_opendd_velocity_calibration import apply, pick, prepare


def test_common_support_and_path_agreement_bound():
    cv = np.zeros((3,12,2))
    ols = cv.copy()
    ols[1,:,0] = 1.
    new = cv.copy()
    new[0,:,0] = 2.
    new[1,:,0] = .2
    valid, cap = prepare(dict(cv=cv,ols=ols,legal=np.ones((3,1),dtype=bool)),new)
    assert valid.tolist()==[True,True,False]
    np.testing.assert_allclose(cap,[.025,1.,1.])
    p = apply(cv,new,.25*valid*cap)
    np.testing.assert_allclose(p[:,0,0],[.0125,.05,0.])


def test_calibration_rejects_positive_harm_even_when_signed_change_is_small():
    def trial(alpha, gain, harm):
        return dict(alpha=alpha,sites=[dict(lower_pp=gain,positive_easy_pct=harm,
                                           signed_easy_pct=-1.)]*6)

    rows = [trial(.025,1.,1.),trial(.05,1.,1.5),trial(.075,2.,2.01),trial(.1,2.,None)]
    assert pick(rows)==.025
    assert pick(rows[-2:]) is None
