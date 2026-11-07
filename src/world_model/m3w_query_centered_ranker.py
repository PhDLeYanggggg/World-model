"""Learn relative action preference without fitting between-query offsets."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors


def reward(bank, cv, target, valid, scale, easy_cut):
    ref,_ = errors(cv,target,valid)
    known = valid.any(1)
    ref = np.where(known,ref,0.)
    easy = known & (ref>0.) & (ref<=easy_cut)
    columns = []
    for k in range(bank.shape[1]):
        path = bank[:,k]
        err,_ = errors(path,target,valid)
        radius = np.linalg.norm(path-cv,axis=-1).max(1)
        missing = np.where(known,0.,radius)
        gain = np.where(known,ref-err,0.)
        harm = np.maximum(-gain,0.)
        u = gain-missing
        a = harm+missing-.02*ref
        e = easy*harm+missing-.02*easy*ref
        columns.append((u-a-e)/scale)
    return np.stack(columns,axis=1)


def moments(x, y, allowed, query):
    """Block family slopes; centering equals a weighted within-query pair loss."""
    n,k,d = x.shape
    if y.shape!=(n,k) or allowed.shape!=(n,k) or query.shape!=(n,):
        raise ValueError('Aligned actions, masks and current queries required')
    _,q = np.unique(query,return_inverse=True)
    sizes = np.bincount(q,weights=allowed.sum(1))
    have = sizes>0
    nq = have.sum()
    if not nq:
        raise ValueError('A fitting source must contain legal actions')
    mean_x = np.zeros((len(sizes),k,d))
    mean_y = np.zeros(len(sizes))
    g,b = np.zeros((k*d,k*d)),np.zeros(k*d)
    for a in range(k):
        mask = allowed[:,a]
        z,t,q0 = x[mask,a],y[mask,a],q[mask]
        w = 1./sizes[q0]
        ss = slice(a*d,(a+1)*d)
        g[ss,ss] = z.T @ (w[:,None]*z)
        b[ss] = z.T @ (w*t)
        np.add.at(mean_x[:,a],q0,w[:,None]*z)
        np.add.at(mean_y,q0,w*t)
    z = mean_x[have].reshape(nq,k*d)
    return (g/nq,b/nq),((g-z.T@z)/nq,(b-z.T@mean_y[have])/nq)


def fit(source_moments, k, ridge=.002):
    g,b = (np.mean(v,axis=0) for v in zip(*source_moments))
    return np.linalg.solve(g+ridge*np.eye(len(b)),b).reshape(k,-1)


def score(x, weights):
    return np.einsum('nak,ak->na',x,weights)
