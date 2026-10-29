import numpy as np
import torch
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_european_source_forecast import fit_design
from src.world_model.m3w_native_forecast import fit_trial as native_fit
from src.world_model.m3w_reference_harm_forecast import fit_trial, assert_training_equivalent


def test_current_producer_neutral_and_resume_exact(tmp_path):
    torch.set_num_threads(4)
    g = np.zeros((12,476),np.float32)
    g[:,:16] = np.arange(-7,1).repeat(2)
    g[:,16:24] = np.arange(-7,1)/12
    g[:,332:356] = np.arange(1,13).repeat(2)
    g[:,38:166] = np.tile(np.arange(-7,1).repeat(2),(12,8))+2
    g[:,166:230] = np.tile(np.arange(-7,1)/12,8)
    g[:,230:294] = 1
    g[:,230:234] = 0
    data=dict(geometry=g,target=g[:,332:356].reshape(-1,12,2)+1,
              valid=np.ones((12,12),bool),sites=np.repeat(['a','b','c'],4))
    fold=fit_design(data,['a','b'],np.tile([3.,2.,1.,2.,2.,2.],(12,1)))
    settings=dict(steps=6,batch_size=4,learning_rate=.001,minimum_lr_ratio=.01,
                  weight_decay=.0001,gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    for arm in ('native','neutral','harm','resumed'):
        torch.manual_seed(17)
        model=DimensionlessAgentTrackSourceForecaster(2,width=16,heads=2,layers=2)
        kw=dict(seed=17,settings=settings,identity={'fixture':True},directory=tmp_path/arm,heartbeat=lambda **kw:None)
        if arm=='native':native_fit(model,data,fold,**kw)
        else:
            w=dict(selected_weight=0.,cv_weight=0.) if arm=='neutral' else {}
            fit_trial(model,data,fold,**kw,**w,stop_at=3 if arm=='resumed' else None)
            if arm=='resumed':
                fit_trial(DimensionlessAgentTrackSourceForecaster(2,width=16,heads=2,layers=2),data,fold,**kw,resume=True)
    load=lambda arm:torch.load(tmp_path/arm/'checkpoint.pt',weights_only=False)
    assert assert_training_equivalent(load('native'),load('neutral'))
    assert assert_training_equivalent(load('harm'),load('resumed'))
