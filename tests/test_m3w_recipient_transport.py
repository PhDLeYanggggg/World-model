import numpy as np
import pytest
import torch

from src.world_model import m3w_recipient_transport as model


def example():
    h = np.arange(-7., 1.)[None, :, None] * np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., 0.]]])
    cv = np.zeros((4, 12, 2))
    anchor = cv.copy()
    anchor[0, :, 0] = np.arange(1., 13.)
    anchor[1, :, 1] = 2.
    return h, np.zeros((4, 2)), np.ones((4, 5)), cv, anchor


def test_transport_preserves_stepwise_norms_and_rotates_heading():
    h, offsets, design, cv, anchor = example()
    drop, add, new, x = model.proposals(h, offsets, design, cv, anchor, [0, 1])
    np.testing.assert_array_equal(drop, [0, 1])
    np.testing.assert_array_equal(add, [2, 2])
    np.testing.assert_allclose(new[:, :, 0], -anchor[drop].max(2))
    np.testing.assert_allclose(new[:, :, 1], 0.)
    assert x.shape == (2, 44)


def test_all_missing_swap_has_minus_one_normalized_gain():
    h, offsets, design, cv, anchor = example()
    drop, add, new, _ = model.proposals(h, offsets, design, cv, anchor, [0, 1])
    y, valid = cv.copy(), np.zeros(cv.shape[:2], dtype=bool)
    np.testing.assert_allclose(model.targets(cv, anchor, y, valid, drop, add, new), -1.)
    y[add[0]] = new[0]
    valid[:] = True
    assert model.targets(cv, anchor, y, valid, drop, add, new)[0] == 1.


def test_no_swap_for_empty_and_nonpositive_and_reject_invalid_recipients():
    assert model.decision(np.zeros(3)) == model.decision(np.empty(0)) == -1
    assert model.decision(np.array([.4, .4, .3])) == 0
    h, offsets, design, cv, anchor = example()
    with pytest.raises(ValueError, match='distinct'):
        model.proposals(h, offsets, design, cv, anchor, [0, 0])
    a, b, new, x = model.proposals(h[:2], offsets[:2], design[:2], cv[:2], anchor[:2], [0, 1])
    assert len(a) == len(b) == len(new) == len(x) == 0
    assert model.network(x.shape[1])(torch.from_numpy(x)).shape == (0,)


def test_robust_loss_concentrates_gradient_on_worse_donor():
    scores = torch.tensor([[.3, .3], [-.3, -.3]], requires_grad=True)
    target = torch.ones_like(scores)
    mean = model.objective(scores, target, False)
    robust = model.objective(scores, target, True)
    mean.backward(retain_graph=True)
    assert scores.grad[0].abs().mean() == scores.grad[1].abs().mean()
    scores.grad.zero_()
    robust.backward()
    assert robust > mean
    assert scores.grad[1].abs().mean() > scores.grad[0].abs().mean()


def test_runner_proposes_without_targets_or_future_availability():
    from scripts import run_m3w_opendd_recipient_transport as run

    h, offsets, design, cv, anchor = example()
    d = dict(history=h, offsets=offsets, x=design, counts=[len(h)])
    packet = dict(cv=cv, bounds=[[0, len(h)]], additive_rows=np.array([[0, 1]]),
                  additive_paths=anchor[:2][None])

    class First(torch.nn.Module):
        def forward(self, x):
            return torch.arange(len(x), 0, -1, dtype=x.dtype)

    np.testing.assert_array_equal(run.propose(packet, d, {'mean': First()})['mean'], [0])


def test_runner_rejects_a_packet_with_leaked_donor_or_held_source():
    from scripts import run_m3w_opendd_recipient_transport as run

    held, donor, *rest = run.base.SITES
    cell = dict(packets={held: {}, donor: {}}, fit_sites=rest)
    run.check_cell(cell)
    for bad in (held, donor):
        with pytest.raises(ValueError, match='exclude both'):
            run.check_cell(dict(cell, fit_sites=[bad, *rest[:-1]]))


def test_ceiling_risk_arithmetic_matches_the_unchanged_recording_policy():
    from scripts.audit_m3w_recipient_transport_ceiling import safe
    from src.evaluation.m3w_recording_switch_ceiling import moments, policy

    floor = np.zeros((3, 12, 2))
    future = floor.copy()
    future[:, :, 0] = np.array([.5, 2., .1])[:, None]
    observed = np.ones((3, 12), dtype=bool)
    for easy_delta, missing in ((.1, False), (-.1, False), (.1, True)):
        pred = floor.copy()
        pred[:2, :, 0] = np.array([easy_delta, .4])[:, None]
        mask = observed.copy()
        mask[0] = not missing
        exact = policy(pred, floor, floor, future, mask, np.array(['site']*3), easy_cut=1., hard_cut=2.)
        full = moments(floor, pred, future, mask, 1.)
        m = {k: v[None, :2] for k, v in full.items()}
        assert bool(safe(m, .02*full['ref_easy'].sum())[0]) == exact['recordings']['site']['supported']
