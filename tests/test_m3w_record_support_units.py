import torch
from src.world_model import m3w_multiaction_record_policy as base
from src.world_model.m3w_record_support_units import support_scaled_pass, installed
from tests.test_m3w_multiaction_record_policy import data_fixture


def test_support_units_change_neither_forward_actions_nor_risk_signs():
    torch.set_num_threads(1); data = data_fixture(); model = base.RecordActionPolicy(8, 17)
    dual = torch.ones((2, 3)); original = base.full_pass
    old = original(model, data, dual, .5, 4)
    new = support_scaled_pass(original, model, data, dual, .5, 4)
    assert old['actions'] == new['actions']
    for key in ('gain', 'risk', 'support'):
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    torch.testing.assert_close(old['excess'][:, :2], new['excess'][:, :2], rtol=0, atol=0)
    assert torch.equal(old['excess'][:, 2] > 0, new['excess'][:, 2] > 0)


def test_empty_support_has_unit_violation_not_inverse_record_size():
    torch.set_num_threads(1); data = data_fixture(); data['anchor'].zero_()
    model = base.RecordActionPolicy(8, 17); dual = torch.ones((2, 3))
    old = base.full_pass(model, data, dual, .1, 4)
    new = support_scaled_pass(base.full_pass, model, data, dual, .1, 4)
    assert old['actions'] == new['actions'] == 0
    torch.testing.assert_close(new['excess'][:, 2], torch.ones(2))
    assert (old['excess'][:, 2] < 1).all()


def test_scoped_override_always_restores_frozen_core():
    original = base.full_pass
    with installed():
        assert base.full_pass is not original
    assert base.full_pass is original
    with installed(False):
        assert base.full_pass is original
