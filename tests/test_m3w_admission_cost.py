import pytest
import torch

from src.world_model import m3w_admission_cost as api
from src.world_model import m3w_budget_score_cost as parent
from src.world_model import m3w_active_bank_cost as active
from tests.test_m3w_active_bank_cost import packets_and_data
from tests.test_m3w_budget_score_cost import settings


def example():
    p = torch.zeros((3, 7, 5))
    p[..., 0] = .1
    p[..., 1] = p[..., 4] = .001
    p[..., 2] = p[..., 3] = 1.
    p[:, 6, 0] = .2
    y = p.clone()
    y[0, 6, 1] = y[0, 6, 4] = .1
    y[0, 0, 1] = y[0, 0, 4] = 100.
    real = torch.ones((3, 7), dtype=torch.bool)
    real[2] = False
    y[2] = 1000.
    return p.requires_grad_(), y, real


def test_only_current_chosen_false_admission_has_safety_gradient():
    p, y, real = example()
    value, info = api.admission_penalty(p, y, real, torch.ones(5))
    assert value > 0 and info['selected_supervised_rows'] == 2
    assert info['false_admitted_channels'] == 2
    value.backward()
    assert p.grad[0, 6, 1] < 0 and p.grad[0, 6, 2] > 0
    assert p.grad[0, 6, 4] < 0 and p.grad[0, 6, 3] > 0
    assert not torch.count_nonzero(p.grad[0, :6])
    assert not torch.count_nonzero(p.grad[1:])
    assert not y.requires_grad


def test_selection_updates_with_prediction_not_fixed_teacher_or_label():
    p, y, real = example()
    before, _ = api.admission_penalty(p, y, real, torch.ones(5))
    with torch.no_grad():
        p[0, 6, 1] = .5
        p[0, 0, 0] = 0.
    after, info = api.admission_penalty(p, y, real, torch.ones(5))
    assert before > 0 and after == 0
    assert info['false_admitted_channels'] == 0
    after.backward()
    assert not torch.count_nonzero(p.grad)


def test_unknown_and_inactive_supervision_cannot_change_penalty():
    p, y, real = example()
    a, _ = api.admission_penalty(p, y, real, torch.ones(5))
    y[2] *= 20
    y[0, :6] *= 30
    b, _ = api.admission_penalty(p, y, real, torch.ones(5))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        api.admission_penalty(p, y, real, torch.zeros(5))


def test_original_and_upper_full_state_controls_and_interrupted_context(tmp_path):
    data, _, _ = packets_and_data()
    kw = dict(variant='new_neural', settings=settings(), seed=17, identity=dict(fixture=True))
    for arm in ('original', 'upper_score'):
        reference = parent.fit(data, objective_arm=arm, **kw, path=tmp_path / (arm + '_old.gz'))
        candidate = api.fit(data, objective_arm=arm, **kw, path=tmp_path / (arm + '.gz'))
        active.assert_scientific_equal(reference, candidate)
    old = active.completion.base.continuous_loss
    with pytest.raises(RuntimeError):
        with api.installed('admitted_score'):
            raise RuntimeError('restore')
    assert active.completion.base.continuous_loss is old


def test_real_torch_admission_objective_resume_preserves_original_draws(tmp_path):
    data, _, _ = packets_and_data()
    kw = dict(variant='new_neural', settings=settings(), seed=17, identity=dict(fixture=True))
    original = api.fit(data, objective_arm='original', **kw, path=tmp_path / 'old.gz')
    direct = api.fit(data, objective_arm='admitted_score', **kw, path=tmp_path / 'direct.gz')
    api.fit(data, objective_arm='admitted_score', **kw, path=tmp_path / 'resume.gz', stop_at=3)
    resumed = api.fit(data, objective_arm='admitted_score', **kw, path=tmp_path / 'resume.gz', resume=True)
    active.assert_scientific_equal(direct, resumed)
    active.assert_common_design(original, direct)
    assert direct['unknown_rows_with_loss'] == 0 and direct['parameters'] == 30405
