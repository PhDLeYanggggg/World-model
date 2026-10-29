import copy
import inspect

import numpy as np
import torch

from src.world_model import m3w_detector_cost as model
from src.world_model import m3w_completion_cost as old
from src.world_model.m3w_detector_history_features import features


def data():
    rng=np.random.default_rng(2);n=20
    z=torch.from_numpy(rng.normal(size=(n,7,398)).astype(np.float32))
    truth=torch.zeros(n,7,5);truth[:]=torch.tensor([.2,.1,1.,.3,.02])
    return dict(z=z,envelope=torch.ones(n,7),truth=truth,bound_truth=truth,known=torch.ones(n,dtype=torch.bool),
        allowed=torch.ones(n,7,dtype=torch.bool),groups=[np.arange(10),np.arange(10,20)],
        preprocess=dict(mean=np.zeros(398),std=np.ones(398),clip=8.,bias=np.zeros(5,np.float32),rms=np.ones(5,np.float32)),
        input_hashes=dict(toy='same'),draw_pool=dict(rows=20,known_rows=20,unknown_rows=0))


SETTINGS=dict(width=64,learning_rate=.0003,batch_size=8,steps=20,underharm_weight=4.,ranking_weight=.5,
              gradient_clip=5.,heartbeat_every=10,checkpoint_every=10)


def test_current_representation_matches_repeated_last_boxes_and_confidence():
    rng=np.random.default_rng(3);boxes=np.zeros((4,8,4));boxes[...,:2]=rng.normal(size=(4,8,2));boxes[...,2:]=boxes[...,:2]+[10,30]
    confidence=rng.uniform(0,1,(4,8));h=features(boxes,confidence)
    keys=np.arange(4)
    np.testing.assert_array_equal(model.representation(h,keys,keys,keys,'current'),features(boxes,confidence,'current'))


def test_disabled_new_branch_reproduces_old_model_optimizer_draws_and_trace(tmp_path):
    torch.set_num_threads(1);base=data();sites=np.array(['a']*10+['b']*10)
    oldstate=old.fit(copy.deepcopy(base),mode='known_only',bank='physical_neural',settings=SETTINGS,seed=17,
                     identity=dict(toy=True),path=tmp_path/'old.pt.gz')
    new=model.training_data(base,np.zeros((20,64)),sites,'disabled')
    with model.installed('disabled'):
        state=old.fit(new,mode='known_only',bank='physical_neural',settings=SETTINGS,seed=17,
                      identity=dict(toy=True),path=tmp_path/'new.pt.gz')
    model.assert_neutral(oldstate,state)


def test_active_history_branch_has_real_updates_and_exact_resume(tmp_path):
    base=data();sites=np.array(['a']*10+['b']*10);extra=np.random.default_rng(4).normal(size=(20,64))
    new=model.training_data(base,extra,sites,'history')
    kw=dict(mode='known_only',bank='physical_neural',settings=SETTINGS,seed=17,identity=dict(toy=True))
    with model.installed('history'):
        old.fit(new,**kw,path=tmp_path/'resume.pt.gz',stop_at=10)
        a=old.fit(new,**kw,path=tmp_path/'resume.pt.gz',resume=True)
        b=old.fit(new,**kw,path=tmp_path/'direct.pt.gz')
    for key in a:
        if key!='seconds':old.base.exact(a[key],b[key])
    assert torch.count_nonzero(a['model']['detector_weight'])>0
    assert 'future' not in str(inspect.signature(model.predict)) and 'known' not in inspect.signature(model.normalize).parameters
