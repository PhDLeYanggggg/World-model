import numpy as np
import torch

from src.world_model import m3w_recipient_realized_value as previous
from src.world_model import m3w_swap_observation_support as model


def test_factorization_reconstructs_direct_conservative_swap_values():
    rng = np.random.default_rng(63)
    cv, anchor, y = rng.normal(size=(3, 5, 12, 2))
    a, b = np.array([0, 1]), np.array([2, 3])
    new = rng.normal(size=(2, 12, 2))
    valid = rng.random((5, 12)) > .2
    _, _, g = previous.geometry(cv, anchor, a, b, new, np.zeros((2, 1)))
    terms = model.targets(y, valid, a, b, g)
    ref = previous.targets(y, valid, a, b, g)
    np.testing.assert_allclose(model.lower(terms), ref[:, [0, 2]]-ref[:, [1, 3]], rtol=0., atol=1e-7)


def test_no_support_masks_conditional_loss_and_has_minus_one_lower_bound():
    y = torch.zeros((2, 2, 3))
    logits = torch.zeros((2, 2, 4), requires_grad=True)
    model.loss(logits, y).backward()
    torch.testing.assert_close(model.lower(y), -torch.ones((2, 2)))
    assert torch.count_nonzero(logits.grad[..., 1:3]) == 0
    assert torch.count_nonzero(logits.grad[..., [0, 3]]) > 0


def test_footprint_is_translation_invariant_and_uses_no_outcomes():
    rng = np.random.default_rng(3)
    h, o, cv = rng.normal(size=(4, 8, 2)), rng.normal(size=(4, 2)), rng.normal(size=(4, 12, 2))
    a, b = np.array([0, 1]), np.array([2, 3])
    x = model.footprint(h, o, cv, a, b)
    np.testing.assert_allclose(x, model.footprint(h, o+[4., 2.], cv, a, b), atol=1e-6)
    assert x.shape == (2, 9)


def test_allocation_changes_only_the_mask_at_fixed_count_and_movement():
    from scripts.run_m3w_opendd_swap_support_allocation import nominate

    old = dict(actions=np.array([0, 1, 2, -1]), cost=np.array([1., 1., 1., 0.]),
               budget_anchor=np.array([True, False, False, False]),
               opportunity=np.array([True, False, False, False]))
    scores = np.zeros((3, 2, 4))
    scores[:, 0, 3] = [.6, .5, .3]
    scores[:, 1, 3] = [.0, .6, 1.]
    result, status = nominate(old, {'footprint': scores})
    np.testing.assert_array_equal(result['actions'], old['actions'])
    assert result['trajectory'].tolist() == [True, False, False, False]
    assert result['joint'].sum() == 1
    assert result['joint'].tolist() != result['trajectory'].tolist()
    assert status['joint'] == 'optimal'
