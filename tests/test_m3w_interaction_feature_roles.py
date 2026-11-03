import numpy as np
import pytest

from scripts import run_m3w_interaction_feature_cost as runner


def setup(monkeypatch, wrong_role=False):
    calls, loaded, supervised = [], [], []
    original_load = np.load
    def parent_bank(job, source, ids, refs, training=False):
        calls.append(training)
        use_train = training and not wrong_role
        chosen = np.array([0,1] if use_train else [2,3])
        site = 'fit' if use_train else 'held'
        c = dict(x=np.zeros((2,7,444),np.float32),envelope=np.ones((2,7)),scales=np.ones(2),
                 allowed=np.ones((2,7),bool),groups=[np.array([0,1])],source_groups=[np.array([0])],
                 sites=np.array([site]*2),ids=chosen)
        return dict(ids=chosen,paths=np.zeros((2,7,12,2)),causal=c)
    def load(path,*args,**kwargs):
        path = str(path)
        loaded.append(path)
        if path.endswith('cached_ids.npy'):
            return np.arange(4)
        if path.endswith('hidden.npy'):
            return np.ones((4,64),np.float32)
        if path.endswith('target_eval.npy'):
            return np.repeat(np.arange(4)[:,None,None],24,1).reshape(4,12,2).astype(float)
        if path.endswith('valid.npy'):
            return np.ones((4,12),bool)
        return original_load(path,*args,**kwargs)
    def training_data(c, paths, target, valid, floor, cut):
        supervised.append((c['ids'].copy(),target[:,0,0].copy()))
        return dict(preprocess=dict(source_names=sorted(set(c['sites']))))
    monkeypatch.setattr(runner.parent,'bank',parent_bank)
    monkeypatch.setattr(runner,'checked',lambda ref:ref['path'])
    monkeypatch.setattr(runner.np,'load',load)
    monkeypatch.setattr(runner.api.base,'training_data',training_data)
    job = dict(outer_trial='one',critic_sites=['fit'],evaluation_sites=['held'],floor_index=1,easy_cut=2.)
    source = dict(origin=np.zeros((4,2)))
    caches = {'one':dict(ids={'path':'cached_ids.npy'},latent={'path':'hidden.npy'})}
    return job,source,caches,calls,loaded,supervised


def test_training_role_reaches_parent_and_only_fit_targets_enter_preprocessing(monkeypatch):
    job,source,caches,calls,loaded,supervised = setup(monkeypatch)
    runner.bank(job,source,np.arange(4),{},caches,training=True)
    assert calls == [True]
    np.testing.assert_array_equal(supervised[0][0],[0,1])
    np.testing.assert_array_equal(supervised[0][1],[0,1])
    loaded.clear()
    runner.bank(job,source,np.arange(4),{},caches,training=False)
    assert calls == [True,False] and len(supervised)==1
    assert not any(p.endswith(('target_eval.npy','valid.npy')) for p in loaded)


def test_wrong_parent_role_is_refused_before_new_target_or_latent_load(monkeypatch):
    job,source,caches,calls,loaded,supervised = setup(monkeypatch,wrong_role=True)
    with pytest.raises(ValueError,match='role must match'):
        runner.bank(job,source,np.arange(4),{},caches,training=True)
    assert calls == [True] and loaded == [] and supervised == []
