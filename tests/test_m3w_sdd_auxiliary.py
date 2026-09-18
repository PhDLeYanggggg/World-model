import copy

import numpy as np
import pytest
import torch

from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_sdd_auxiliary import (
    AuxiliaryGeometry, fit_two_phase, masked_log_loss, predict, validate_registration,
)


def registration():
    return dict(approval={'user_reply': '按这个方案继续', 'date': '2026-09-18'},
        data_role='supervised_auxiliary_training', original_split='train',
        original_train_recordings=40, raw_frame_stride=12, observed_steps=8,
        predicted_steps=12, main_role='fit_only_exploratory', main_primary_changed=False,
        modalities=['geometry', 'mask_only', 'past_rgb'], schedules=['no_aux', 'sdd_aux'])


@pytest.mark.parametrize('field,value', [('original_split','test'),('raw_frame_stride',1),
    ('data_role','diagnostic_only'),('main_primary_changed',True),('original_train_recordings',60)])
def test_reject_unapproved_admission(field, value):
    reg = registration(); reg[field] = value
    with pytest.raises(ValueError):
        validate_registration(reg)


def test_wrapper_keeps_future_out_and_diagnostic_source_role_unchanged():
    rows = np.array([[0,f,0,f+4,6,f,0,0,1] for f in range(300)], float)
    labels = np.array(['Pedestrian']*len(rows))
    roster = [f'train/video{i}' for i in range(40)]
    reg = registration()
    with pytest.raises(ValueError):
        AuxiliaryGeometry(rows, labels, 'test/video0', reg, roster)
    original = AuxiliaryGeometry(rows, labels, roster[0], reg, roster)
    before = original.inputs(2)
    changed = rows.copy(); changed[changed[:,5] > 108, 1:5] += 10000
    after = AuxiliaryGeometry(changed, labels, roster[0], reg, roster).inputs(2)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert original.adapter.metadata['data_role'] == 'diagnostic_only'


def test_masked_log_loss_has_no_gradient_from_missing_targets():
    pred = torch.ones(3,12,2,requires_grad=True)
    target = torch.full_like(pred,float('nan')); target[0,:3] = 0
    valid = torch.zeros(3,12,dtype=torch.bool); valid[0,:3] = True
    loss, detail = masked_log_loss(pred,target,valid); loss.backward()
    assert detail['supported_rows'] == 1 and torch.isfinite(loss)
    assert not pred.grad[~valid].any()
    empty, detail = masked_log_loss(pred,target,valid & False)
    assert empty.item() == 0 and detail['masked_ADE'] is None


def payload():
    torch.manual_seed(55)
    return dict(geometry=torch.randn(7,9),observed=torch.rand(7,8),
        baseline=torch.randn(7,12,2),target=torch.randn(7,12,2),
        valid=torch.ones(7,12,dtype=torch.bool),
        rgb=torch.rand(7,8,3,32,32),coverage=torch.ones(7,8,1,32,32))


def test_inference_ignores_loss_targets_and_future_fields():
    batch = payload(); model = OfflineVisualForecast(9)
    with torch.no_grad():
        model.output.weight.fill_(.1)
    for modality in ('geometry','mask_only','past_rgb'):
        expected = predict(model,batch,modality)
        changed = copy.copy(batch)
        changed.update(target=torch.full_like(batch['target'],float('nan')),
                       valid=~batch['valid'],future_endpoint=torch.ones(7,2)*1e9)
        assert torch.equal(expected,predict(model,changed,modality))


@pytest.mark.parametrize('stop', [2,3,4])
def test_exact_resume_before_at_after_phase_boundary(tmp_path, stop):
    torch.set_num_threads(2); values = payload()
    config = dict(pretraining_updates=3,main_updates=4,batch_size=3,
        learning_rate=.001,weight_decay=.001,checkpoint_every=2)
    def run(name, stop_at=None):
        torch.manual_seed(17); model = OfflineVisualForecast(9)
        batch = lambda ids:{k:v[ids] for k,v in values.items()}
        result = fit_two_phase(model,batch,batch,7,7,modality='geometry',schedule='sdd_aux',
            config=config,seed=17,identity={'test':1},checkpoint=tmp_path/name,
            heartbeat=lambda _:None,stop_at=stop_at)
        return model,result
    full, _ = run('full.pt')
    run('resumed.pt',stop)
    resumed, result = run('resumed.pt')
    assert result['step'] == 7 and result['new_updates_this_invocation'] == 7-stop
    for k,v in full.state_dict().items():
        assert torch.equal(v,resumed.state_dict()[k])
    _, done = run('resumed.pt')
    assert done['new_updates_this_invocation'] == 0


def test_same_main_stream_and_no_auxiliary_access_in_control(tmp_path):
    torch.set_num_threads(2); values = payload()
    config = dict(pretraining_updates=3,main_updates=4,batch_size=3,
        learning_rate=.001,weight_decay=.001,checkpoint_every=2)
    streams = {}
    for schedule in ('no_aux','sdd_aux'):
        main_calls, aux_calls = [], []
        def batch(ids, calls):
            calls.append(ids.copy())
            return {k:v[ids] for k,v in values.items()}
        torch.manual_seed(17); model = OfflineVisualForecast(9)
        fit_two_phase(model,lambda ids:batch(ids,main_calls),lambda ids:batch(ids,aux_calls),
            7,7,modality='geometry',schedule=schedule,config=config,seed=17,
            identity={'test':schedule},checkpoint=tmp_path/(schedule+'.pt'),heartbeat=lambda _:None)
        streams[schedule] = main_calls[-4:]
        assert len(aux_calls) == (3 if schedule == 'sdd_aux' else 0)
    np.testing.assert_array_equal(streams['no_aux'],streams['sdd_aux'])
