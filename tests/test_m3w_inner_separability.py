import numpy as np
import pytest
import torch
from src.world_model import m3w_inner_separability as api


def toy():
    rng = np.random.default_rng(23); n = 40
    x = rng.normal(size=(n, 6)); env = np.ones(n)*2
    cv = np.linspace(.1, 1, n); floor = cv*.9
    neural = floor+np.where(x[:, 0] > 0, -.05, .08)
    y = api.targets(cv, floor, neural, .5)
    sites = np.array(['train']*n); recordings = np.array(['r']*n); frames = np.arange(n)//4
    pr = api.preprocess(x, env, y, sites, recordings, frames, training_site='train')
    return x, env, y, sites, recordings, frames, pr


def settings():
    return dict(steps=8, width=8, learning_rate=.001, query_batch_size=4,
                gradient_clip=5., checkpoint_every=4, heartbeat_every=4)


def test_only_inner_training_source_can_fit_preprocessing():
    x, e, y, s, r, f, _ = toy(); s[-1] = 'test'
    with pytest.raises(ValueError): api.preprocess(x, e, y, s, r, f, training_site='train')


def test_decoder_bounds_and_equal_initial_predictions():
    x, e, _, _, _, _, pr = toy()
    values = [api.initialize(pr, arm, 8, 17)(torch.zeros((len(x), 6)), torch.tensor(e, dtype=torch.float32)) for arm in api.ARMS]
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    for p in values:
        assert torch.all(p >= 0)
        assert torch.all(p[:, 0]+p[:, 1] <= torch.tensor(e)+1e-6)
        assert torch.all(p[:, 3] <= p[:, 2]) and torch.all(p[:, 4] <= p[:, 1])


def test_training_resume_and_sampling_match(tmp_path):
    x, e, y, s, r, f, pr = toy(); kw = dict(settings=settings(), identity={'source':'train'}, seed=17, heartbeat=lambda **_:None)
    states = {}
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz')
        api.fit(x, e, y, s, r, f, pr, arm=arm, path=path, stop_at=4, **kw)
        states[arm] = api.fit(x, e, y, s, r, f, pr, arm=arm, path=path, resume=True, **kw)
        direct = api.fit(x, e, y, s, r, f, pr, arm=arm, path=tmp_path/(arm+'fresh.pt.gz'), **kw)
        for k in direct:
            if k != 'seconds': api.exact(states[arm][k], direct[k])
        assert states[arm]['step'] == 8
        assert states[arm]['unknown_rows_sampled'] == 0
    api.assert_matched(states['affine'], states['nonlinear'])


def test_unknown_labels_do_not_enter_training(tmp_path):
    x, e, y, s, r, f, _ = toy(); y[-4:] = np.nan
    pr = api.preprocess(x, e, y, s, r, f, training_site='train')
    assert pr['known_rows'] == 36
    st = api.fit(x,e,y,s,r,f,pr,arm='affine',settings=settings(),identity={},seed=17,
                 path=tmp_path/'a.pt.gz',heartbeat=lambda **_:None)
    assert st['unknown_rows_sampled'] == 0
    y[-1, 0] = 0
    with pytest.raises(ValueError): api.preprocess(x,e,y,s,r,f,training_site='train')


def test_action_matching_and_tie_breaks():
    ids = np.array([3,1,2,4]); rec = np.array(['r']*4); frames = np.array([0,0,0,1])
    p = np.array([[1.,0.,1.,.5,0.]]*4)
    other=p.copy(); other[2,1]=1.
    a = api.decisions({'affine':p,'nonlinear':other},np.ones(4,bool),np.ones(4,bool),rec,frames,ids)
    assert a['affine_matched'].tolist() == [False,True,True,True]
    assert a['nonlinear_matched'].tolist() == [True,True,False,True]
    assert a['affine_matched'].sum() == a['nonlinear_matched'].sum()


def test_proper_objective_minimized_at_exact_targets():
    p=torch.tensor([[1.,0.,2.,1.,0.],[0.,1.,2.,1.,.5]])
    seg=torch.tensor([0,0]); rms=torch.ones(8)
    assert api.losses(p,p,seg,1,rms)['total'] == 0
    assert api.losses(p+.1,p,seg,1,rms)['total'] > 0


def test_real_score_entry_rejects_future_fields():
    from scripts.run_m3w_inner_separability import view_predictions
    with pytest.raises(ValueError): view_predictions({}, {'future_endpoint': np.zeros(2)}, 0, 'x', {})


def test_malformed_action_keys_do_not_broadcast():
    p = np.ones((3,5))
    with pytest.raises(ValueError): api.decisions(dict(affine=p,nonlinear=p),np.ones((3,1),bool),
        np.ones(3,bool),np.array(['r']*3),np.arange(3),np.arange(3))
