import numpy as np

from src.world_model import m3w_query_centered_ranker as m


def test_moments_match_explicit_pair_contrasts_and_masks():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5,3,4))
    y = rng.normal(size=(5,3))
    keep = rng.random((5,3))>.2
    q = np.array([0,0,1,1,2])
    pointwise,centered = m.moments(x,y,keep,q)
    full = np.zeros((5,3,12))
    for j in range(3):
        full[:,j,4*j:4*j+4] = x[:,j]
    gs,bs = [],[]
    for t in np.unique(q):
        z,w = full[keep&(q[:,None]==t)],y[keep&(q[:,None]==t)]
        z1,w1 = z[:,None]-z[None,:],w[:,None]-w[None,:]
        gs.append(np.einsum('ijd,ije->de',z1,z1)/(2*len(z)**2))
        bs.append(np.einsum('ijd,ij->d',z1,w1)/(2*len(z)**2))
    np.testing.assert_allclose(centered[0],np.mean(gs,axis=0),atol=1e-12)
    np.testing.assert_allclose(centered[1],np.mean(bs,axis=0),atol=1e-12)
    before,after = m.moments(x,y+(q[:,None]+1)*100,keep,q)
    np.testing.assert_allclose(centered[1],after[1],atol=1e-12)
    assert not np.allclose(pointwise[1],before[1])


def test_reward_preserves_missing_penalty_and_easy_support():
    cv = np.zeros((3,12,2))
    bank = cv[:,None].copy()
    bank[...,0] += 2.
    y = cv.copy()
    y[:, :, 0] = [.5]
    valid = np.ones((3,12),bool)
    valid[1] = False
    y[2,:,0] = 2.
    r = m.reward(bank,cv,y,valid,np.ones(3),1.)
    np.testing.assert_allclose(r.ravel(),[-2.98,-6.,2.04])
