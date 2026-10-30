import numpy as np
import pytest
import torch

from src.world_model import m3w_active_bank_cost as active
from src.world_model import m3w_budget_score_cost as api
from tests.test_m3w_active_bank_cost import packets_and_data


def settings():
    return dict(width=64, steps=6, batch_size=8, learning_rate=.001,
        underharm_weight=4., ranking_weight=.5, gradient_clip=5.,
        heartbeat_every=2, checkpoint_every=2)


def test_signed_score_uses_original_budget_and_corrects_reference_inflation():
    truth = torch.tensor([[[.2, .1, 1., .5, .05]]])
    prediction = truth.clone().requires_grad_()
    with torch.no_grad():
        prediction[..., 2] += 2.
    real = torch.ones((1, 1), dtype=torch.bool)
    loss = api.score_penalty(prediction, truth, real, torch.ones(5), underprediction_weight=4.)
    loss.backward()
    assert loss > 0
    assert prediction.grad[..., 2] > 0  # Gradient descent reduces the optimistic denominator.
    assert prediction.grad[..., 1] < 0  # Or increases the underestimated risk numerator.
    assert not truth.requires_grad
    with pytest.raises(ValueError):
        api.score_penalty(prediction, truth, real, torch.ones(5), underprediction_weight=.5)


def test_upper_score_penalizes_risk_underprediction_not_equal_size_overprediction():
    y = torch.zeros((1, 1, 5))
    y[..., 1] = y[..., 4] = 1.
    low, high = y.clone(), y.clone()
    low[..., 1] -= .2
    high[..., 1] += .2
    real, rms = torch.ones((1, 1), dtype=torch.bool), torch.ones(5)
    lower = api.score_penalty(low, y, real, rms, underprediction_weight=4.)
    upper = api.score_penalty(high, y, real, rms, underprediction_weight=4.)
    torch.testing.assert_close(lower, 4 * upper)
    torch.testing.assert_close(api.score_penalty(low, y, real, rms, underprediction_weight=1.), upper)


def test_unknown_or_inactive_targets_cannot_enter_score_penalty():
    y = torch.ones((2, 7, 5))
    p = torch.zeros_like(y).requires_grad_()
    active_rows = torch.zeros((2, 7), dtype=torch.bool)
    active_rows[0, 0] = True
    a = api.score_penalty(p, y, active_rows, torch.ones(5), underprediction_weight=4.)
    changed = y.clone()
    changed[1] *= 1000
    changed[0, 1:] *= 2000
    b = api.score_penalty(p, changed, active_rows, torch.ones(5), underprediction_weight=4.)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward()
    assert not torch.count_nonzero(p.grad[~active_rows])
    zero = api.score_penalty(p, y, torch.zeros_like(active_rows), torch.ones(5), underprediction_weight=4.)
    assert zero.item() == 0


def test_original_has_exact_torch_training_state_and_patch_is_always_restored(tmp_path):
    data, _, _ = packets_and_data()
    kw = dict(variant='new_neural', settings=settings(), seed=17, identity=dict(fixture=True))
    reference = active.fit(data, **kw, path=tmp_path / 'reference.gz')
    original = active.completion.base.continuous_loss
    objective = active.completion.OBJECTIVE
    candidate = api.fit(data, objective_arm='original', **kw, path=tmp_path / 'candidate.gz')
    active.assert_scientific_equal(reference, candidate)
    with pytest.raises(RuntimeError):
        with api.installed('upper_score'):
            assert active.completion.base.continuous_loss is not original
            raise RuntimeError('restore module after an interrupted training call')
    assert active.completion.base.continuous_loss is original
    assert active.completion.OBJECTIVE == objective


def test_all_three_real_torch_objectives_resume_exactly_and_keep_sampling(tmp_path):
    data, _, _ = packets_and_data()
    states = []
    for arm in api.OBJECTIVES:
        kw = dict(objective_arm=arm, variant='new_neural', settings=settings(), seed=17,
            identity=dict(fixture=True, objective_arm=arm))
        direct = api.fit(data, **kw, path=tmp_path / (arm + '.gz'))
        api.fit(data, **kw, path=tmp_path / (arm + '_resume.gz'), stop_at=3)
        resumed = api.fit(data, **kw, path=tmp_path / (arm + '_resume.gz'), resume=True)
        active.assert_scientific_equal(direct, resumed)
        states.append(direct)
        assert direct['unknown_rows_with_loss'] == 0
        assert direct['parameters'] == 30405
    for state in states[1:]:
        active.assert_common_design(states[0], state)
    assert states[0]['objective'] != states[1]['objective'] != states[2]['objective']
    assert any(not torch.equal(states[0]['model'][k], states[2]['model'][k]) for k in states[0]['model'])
