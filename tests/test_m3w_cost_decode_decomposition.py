import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_selection_cost_forest as forest
from src.evaluation.m3w_selection_cost_fit_objective import score
from src.evaluation.m3w_cost_decode_decomposition import prediction_stages, augmented_objective


def fixture():
    rng = np.random.default_rng(71); n,d = 8,4
    c = dict(x=rng.normal(size=(n,7,d)).astype(np.float32),envelope=rng.uniform(.5,3,(n,7)),
        scales=np.linspace(.5,2,n).astype(np.float32),allowed=np.ones((n,7),bool),
        groups=[np.arange(3),np.arange(3,7),np.array([7])],source_groups=[np.arange(3)],
        sites=np.repeat('source',n),ids=np.arange(n))
    c['allowed'][:,2] = False
    truth = rng.uniform(.01,1,(n,7,5)).astype(np.float32)
    truth[:,:,2:4] = truth[:,0:1,2:4]
    truth[:,:,3] = np.minimum(truth[:,:,3],truth[:,:,2])
    return dict(causal=c,preprocess=dict(mean=np.zeros(d),std=np.ones(d),clip=8.,
        rms=np.array([.7,.8,1.2,.9,.6],dtype=np.float32)),truth=torch.from_numpy(truth),known=np.ones(n,bool))


@pytest.mark.parametrize('arm',forest.ARMS)
def test_final_stage_matches_frozen_decoder_and_chunking(tmp_path,arm):
    data = fixture(); matrix = forest.training_matrix(data,np.array([4,5,7]),arm)
    state = forest.fit(matrix,preprocess=data['preprocess'],arm=arm,seed=17,
        settings=dict(trees=4,max_depth=3,min_samples_leaf=1,max_features=.5,fit_threads=1,checkpoint_every=2),
        identity={'test':True},path=tmp_path/'model.gz')
    native,aug = prediction_stages(state,data['causal'],chunk_size=3)
    other,other_aug = prediction_stages(state,data['causal'],chunk_size=5)
    np.testing.assert_array_equal(native['projected'],forest.predict(state,data['causal']))
    for stage in native: np.testing.assert_array_equal(native[stage],other[stage])
    np.testing.assert_array_equal(aug,other_aug)
    refs = native['shared_reference'][:,:,2:4]
    np.testing.assert_array_equal(refs,np.repeat(refs[:,0:1],7,axis=1))
    measured = augmented_objective(aug,data['truth'].numpy(),data['causal']['allowed'],data['known'],
        data['preprocess']['rms'],data['causal']['groups'],np.array([4,5,7]))
    assert measured['matrix_hashes']['targets'] == matrix['input_hashes']['targets']
    assert measured['matrix_hashes']['weights'] == matrix['input_hashes']['weights']
    cost = score(native['unconstrained'],data['truth'].numpy(),data['causal']['allowed'],data['known'],
        data['causal']['scales'],data['preprocess']['rms'],data['causal']['groups'],np.array([4,5,7]),data['causal']['sites'])
    assert measured['total'] == pytest.approx(cost['total'],rel=2e-6,abs=1e-7)
    bad = copy.deepcopy(data['causal']); bad['future_endpoint'] = np.zeros((8,2))
    with pytest.raises(ValueError,match='past-only'): prediction_stages(state,bad)


def test_unknown_queries_remain_zero_supervision_not_training_labels():
    truth = np.ones((3,7,5),np.float32); truth[2] = np.nan
    p = forest.augmented_targets(np.ones((21,5),np.float32)*2,np.ones(5)).reshape(3,7,6)
    out = augmented_objective(p,truth,np.ones((3,7),bool),np.array([True,True,False]),np.ones(5),
        [np.array([0,1]),np.array([2])],np.array([4,6]))
    assert out['total'] == pytest.approx(.4,rel=1e-6)
    assert out['zero_supervision_query_visits'] == 6
    assert out['unknown_supervised_rows'] == 0


def test_alias_reference_averaging_can_damage_the_legal_action_objective():
    truth = np.zeros((1,7,5)); truth[:,:,2:4] = 1.
    raw = truth.copy(); raw[:,0,2:4] = 100.
    allowed = np.ones((1,7),bool); allowed[:,0] = False
    args = (truth,allowed,np.ones(1,bool),np.ones(1),np.ones(5),[np.array([0])],np.array([4]),np.array(['s']))
    shared = raw.copy(); shared[:,:,2:4] = shared[:,:,2:4].mean(1,keepdims=True)
    assert score(raw,*args)['total'] == 0.
    assert score(shared,*args)['total'] > 1.


def test_radial_constraint_projection_need_not_improve_the_RMS_objective():
    raw = np.tile([1.,1.,1.,0.,0.],(1,7,1))
    truth = np.tile([0.,1.,1.,0.,0.],(1,7,1))
    projected = forest.project_moments(raw.reshape(7,5),np.ones(7)).reshape(1,7,5)
    args = (truth,np.ones((1,7),bool),np.ones(1,bool),np.ones(1),np.array([10.,.1,1.,1.,1.]),
            [np.array([0])],np.array([4]),np.array(['s']))
    assert score(projected,*args)['total'] > score(raw,*args)['total']
