import numpy as np
import pytest
import torch

from src.world_model.m3w_auxiliary_mechanism import arm_config, residual_donors, permuted_batch
from src.world_model.m3w_sdd_auxiliary import fit_two_phase, predict
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast


def test_donors_preserve_exact_support_and_source_not_fixed_pairs():
    records = np.array([0]*6+[1]*4)
    valid = np.ones((10,12), bool); valid[3:6,6:] = False; valid[-1,:] = False
    donors = residual_donors(records,valid,17)
    np.testing.assert_array_equal(np.sort(donors),np.arange(10))
    np.testing.assert_array_equal(records[donors],records)
    np.testing.assert_array_equal(valid[donors],valid)
    assert not (donors[:-1] == np.arange(9)).any()
    assert donors[-1] == 9
    np.testing.assert_array_equal(donors,residual_donors(records,valid,17))


def test_only_loss_targets_change_and_residuals_keep_their_marginal():
    torch.manual_seed(4)
    baseline = torch.randn(6,12,2); target = torch.randn(6,12,2)
    valid = torch.ones(6,12,dtype=torch.bool)
    source = dict(baseline=baseline.numpy(),target=target.numpy(),valid=valid.numpy())
    batch = dict(geometry=torch.randn(6,9),observed=torch.rand(6,8),baseline=baseline,
        target=target,valid=valid,rgb=torch.rand(6,8,3,32,32),coverage=torch.ones(6,8,1,32,32))
    donors = residual_donors(np.zeros(6,int),valid.numpy(),17)
    changed = permuted_batch(batch,np.arange(6),donors,source)
    for key in batch:
        if key != 'target':
            assert changed[key] is batch[key]
    torch.testing.assert_close(changed['target']-baseline,(target-baseline)[donors])
    model = OfflineVisualForecast(9)
    with torch.no_grad():
        model.output.weight.fill_(.1)
        for modality in ('geometry','mask_only','past_rgb'):
            assert torch.equal(predict(model,batch,modality),predict(model,changed,modality))


def test_unsupported_donors_rejected():
    records = np.zeros(2,int); valid = np.ones((2,12),bool)
    with pytest.raises(ValueError):
        residual_donors(records,valid[:,:8],17)
    valid[1,-1] = False
    batch = dict(baseline=torch.zeros(2,12,2),valid=torch.from_numpy(valid))
    source = dict(valid=valid,baseline=np.zeros((2,12,2)),target=np.zeros((2,12,2)))
    with pytest.raises(ValueError):
        permuted_batch(batch,np.arange(2),np.array([1,0]),source)


@pytest.mark.parametrize('stop',[1,2,3])
def test_zero_pretraining_exact_resume_and_matched_main_stream(tmp_path,stop):
    torch.set_num_threads(2); torch.manual_seed(41)
    values = dict(geometry=torch.randn(7,9),observed=torch.rand(7,8),
        baseline=torch.randn(7,12,2),target=torch.randn(7,12,2),valid=torch.ones(7,12,dtype=torch.bool))
    parent = dict(training=dict(pretraining_updates=3,main_updates=4,batch_size=3,
        learning_rate=.001,weight_decay=.001,checkpoint_every=2))
    config,schedule = arm_config('main4k',parent)
    streams = {}
    def run(name,conf,mode,limit=None):
        calls=[]
        def batch(ids):
            calls.append(ids.copy()); return {k:v[ids] for k,v in values.items()}
        torch.manual_seed(17); model=OfflineVisualForecast(9)
        result=fit_two_phase(model,batch,batch,7,7,modality='geometry',schedule=mode,
            config=conf,seed=17,identity={'case':name},checkpoint=tmp_path/name,
            heartbeat=lambda _:None,stop_at=limit)
        streams[name]=calls
        return model,result
    full,result=run('full.pt',config,schedule)
    assert result['auxiliary_draws']==0 and result['main_draws']==12
    run('resume.pt',config,schedule,stop)
    resumed,result=run('resume.pt',config,schedule)
    assert result['new_updates_this_invocation']==4-stop
    for k,v in full.state_dict().items():
        assert torch.equal(v,resumed.state_dict()[k])
    run('source.pt',parent['training'],'sdd_aux')
    np.testing.assert_array_equal(streams['full.pt'],streams['source.pt'][-4:])
