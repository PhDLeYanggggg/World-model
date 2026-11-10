import numpy as np

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.evaluation.m3w_opendd_location import paired
from src.world_model import m3w_recipient_realized_value as model


def case():
    rng = np.random.default_rng(13)
    cv, anchor, truth = rng.normal(size=(3, 5, 12, 2))
    drop, add = np.array([0, 3, 2]), np.array([1, 4, 1])
    new = rng.normal(size=(len(add), 12, 2))
    valid = rng.random((5, 12)) > .4
    return cv, anchor, truth, drop, add, new, valid


def test_split_benefit_and_charge_reproduce_the_two_full_grid_bounds():
    cv, anchor, truth, a, b, new, valid = case()
    x, scale, g = model.geometry(cv, anchor, a, b, new, np.zeros((len(a), 3)))
    assert x.shape == (3, 6)
    y = model.targets(truth, valid, a, b, g)
    for i, (drop, add) in enumerate(zip(a, b, strict=True)):
        pred = anchor.copy()
        pred[drop], pred[add] = cv[drop], new[i]
        _, individual = paired(anchor, pred, truth, valid)
        _, relative = pair_reduction(pred[None], anchor, truth, valid)[0]
        np.testing.assert_allclose((y[i, [0, 2]]-y[i, [1, 3]])*scale[i],
                                   [individual.mean(), relative], rtol=0., atol=1e-7)


def test_unobserved_outcome_only_charges_the_movement_bound():
    cv, anchor, truth, a, b, new, valid = case()
    _, scale, g = model.geometry(cv, anchor, a, b, new, np.zeros((len(a), 1)))
    y = model.targets(truth, np.zeros_like(valid), a, b, g)
    np.testing.assert_allclose(y, np.tile([0., 1., 0., 1.], (len(a), 1)))
    value, joint = model.utility(y, scale)
    np.testing.assert_allclose(value, -scale[:, 0])
    np.testing.assert_allclose(joint, value-model.RELATIVE_WEIGHT*scale[:, 1])


def test_direct_relative_interval_agrees_with_reversed_bound():
    from scripts.run_m3w_opendd_realized_swap_value import relative_contrast

    pred, ref, y, _, _, _, valid = case()
    seen, lower, upper = relative_contrast(pred, ref, y, valid)
    a, b, c = relative_contrast(ref, pred, y, valid)
    np.testing.assert_allclose([seen, lower, upper], [-a, -c, -b], rtol=0., atol=1e-12)
