import numpy as np

from src.world_model import m3w_full_path_selection as s


def inputs():
    h = np.zeros((4, 8, 2))
    h[..., 0] = np.arange(8)[None]
    h[..., 1] = np.arange(4)[:, None]
    q = np.array([0, 0, 0, 1])
    cv = np.zeros((4, 12, 2))
    p = np.zeros((4, 4, 12, 2))
    p[..., 0] = np.array([1., 2., 3., 9.])[None, :, None]
    return h, q, cv, p


def test_matched_bank_is_independent_of_neural_peaks_and_does_not_amplify():
    _, q, cv, p = inputs()
    _, match = s.banks(cv, p, q)
    np.testing.assert_array_equal(match[..., 0], np.broadcast_to([0., 2., 2., 2.], (12, 4)).T[None]+np.zeros((4, 4, 12)))
    _, small = s.banks(cv, p[:, :2], q)
    np.testing.assert_array_equal(match[:, :2], small)


def test_fixed_controls_match_budgets_but_strict_can_abstain():
    h, q, cv, p = inputs()
    _, p = s.banks(cv, p, q)
    legal = np.linalg.norm(p-cv[:, None], axis=-1).max(2)>0
    value = np.ones((4, 4, 6))
    routes = s.choices(h,q,cv,p,value,legal,fixed=True)
    s.matched(cv,q,routes)
    assert all((r!=cv).any((1,2)).sum()==2 for r in routes.values())
    strict = s.choices(h,q,cv,p,value,legal,fixed=False)
    np.testing.assert_array_equal(strict['joint'],cv)
    value[...,2:4] = 0.
    strict = s.choices(h,q,cv,p,value,legal,fixed=False)
    assert (strict['joint']!=cv).any((1,2)).sum()==2
