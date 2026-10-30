import pytest
import torch
from src.world_model import m3w_gradient_balanced_admission as api
from tests.test_m3w_active_bank_cost import packets_and_data
from tests.test_m3w_budget_score_cost import settings


def test_equalization_is_detached_and_changes_no_accumulated_gradients():
    w = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    anchor, aux = w.square().sum(), 1e-4*w[0]
    total, info = api.balanced_total(anchor, aux, [w])
    assert w.grad is None
    assert info['weighted_auxiliary_over_anchor'] == pytest.approx(1.)
    assert not info['capped']
    total.backward()
    torch.testing.assert_close(w.grad, torch.tensor([2.+info['scale']*1e-4, 4.], dtype=torch.float64))


def test_zero_and_tiny_gradients_do_not_create_nonfinite_weights():
    w = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    anchor = w.square().sum()
    zero, info = api.balanced_total(anchor, w.sum()*0, [w])
    assert info['scale'] == 0 and torch.equal(zero, anchor)
    _, tiny = api.balanced_total(anchor, w[0]*1e-10, [w])
    assert tiny['capped'] and tiny['scale'] == 1e6
    assert 0 < tiny['weighted_auxiliary_over_anchor'] < 1


def test_real_torch_resume_restores_atomic_balance_statistics_and_common_design(tmp_path):
    data, _, _ = packets_and_data()
    kw = dict(variant='new_neural', settings=settings(), seed=17, identity=dict(fixture=True))
    old = api.frozen.fit(data, objective_arm='admitted_score', **kw, path=tmp_path/'unit.gz')
    direct = api.fit(data, **kw, path=tmp_path/'direct.gz')
    api.fit(data, **kw, path=tmp_path/'resume.gz', stop_at=3)
    resumed = api.fit(data, **kw, path=tmp_path/'resume.gz', resume=True)
    api.frozen.parent.active.assert_scientific_equal(direct, resumed)
    api.frozen.parent.active.assert_common_design(old, direct)
    assert direct['gradient_balance']['step'] == 6
    assert direct['unknown_rows_with_loss'] == 0 and direct['parameters'] == 30405


def test_hooks_restore_on_interruption():
    base = api.frozen.parent.active.completion.base
    detector = api.frozen.parent.active.detector
    before = (base.continuous_loss, base.save_checkpoint, detector.DetectorCostHead)
    with pytest.raises(RuntimeError):
        with api.installed(settings()):
            raise RuntimeError('interrupted')
    assert before == (base.continuous_loss, base.save_checkpoint, detector.DetectorCostHead)
