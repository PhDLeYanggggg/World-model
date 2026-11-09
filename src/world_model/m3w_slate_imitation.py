"""Imitate checked recording plans using only query-local information."""
import numpy as np
import torch
from torch import nn

from src.world_model import m3w_reference_bridge_policy as costs


def options(row, reserve):
    ids = np.arange(len(row), dtype=int)
    first, second = np.triu_indices(len(row), k=1)
    allowed = row[first] != row[second]
    out = [np.array([[-1,-1]])]
    if reserve >= 1:
        out.append(np.column_stack((ids, np.full(len(row),-1))))
    if reserve >= 2:
        out.append(np.column_stack((first[allowed], second[allowed])))
    return np.concatenate(out)


def pack(x, center, delta, unit, opt):
    on = opt >= 0
    idx = np.maximum(opt,0)
    a,b = idx[...,0],idx[...,1]
    scale = (unit[a]+unit[b])*.5
    pair = np.column_stack((np.linalg.norm(center[a]-center[b],axis=-1),
        np.linalg.norm(delta[a]-delta[b],axis=-1)[:,[2,5,11]]))
    pair = np.log1p(pair/scale[:,None])
    return (x[idx], pair.astype(np.float32), on)


class Slate(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(costs.WIDTH,48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.unary = nn.Linear(24,1)
        self.pair = nn.Sequential(nn.Linear(52,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x,g,on):
        h = self.action(x)
        a,b = h.unbind(-2)
        single = (self.unary(h).squeeze(-1)*on).sum(-1)
        cross = self.pair(torch.cat((a+b,(a-b).abs(),g),dim=-1)).squeeze(-1)
        return single, single+cross*on.all(-1)


def loss(net,x,g,on,legal):
    a,b = net(x,g,on)
    label = torch.zeros(len(a),dtype=torch.long,device=a.device)
    return sum(nn.functional.cross_entropy(v.masked_fill(~legal,-torch.inf),label) for v in (a,b))*.5


def flat_features(h,cv,ols,bank):
    x,unit,legal,_ = costs.features(h,cv,ols,bank)
    r,k = np.nonzero(legal)
    return r,k,(x[r,k],h[r,-1],bank[r,k]-cv[r],unit[r])


@torch.inference_mode()
def infer(net,h,cv,ols,bank,q):
    r,k,data = flat_features(h,cv,ols,bank)
    p = {k:cv.copy() for k in ('joint','independent','matched_independent','matched_uniform')}
    unsupported = []
    for t in np.unique(q):
        at = np.flatnonzero(q[r]==t)
        keep = np.flatnonzero(q==t)
        if not len(at) or len(keep)<2:
            continue
        opt = options(r[at],min(len(keep)-1,2))
        feat = pack(*(v[at] for v in data),opt)
        solo,joint = [a.numpy() for a in net(*(torch.from_numpy(x) for x in feat))]
        for name, values in (('joint',joint),('independent',solo)):
            use = at[opt[int(values.argmax())][opt[int(values.argmax())]>=0]]
            p[name][r[use]] = bank[r[use],k[use]]
        use = at[opt[int(joint.argmax())][opt[int(joint.argmax())]>=0]]
        if len(use):
            radii = np.linalg.norm(data[2][use],axis=-1).max(1)
            edited = match(net,h[keep],cv[keep],ols[keep],bank[keep],radii)
            p['matched_independent'][keep] = edited[0]
            if edited[1] is None:
                unsupported.append(int(t))
            else:
                p['matched_uniform'][keep] = edited[1]
    return p,np.asarray(unsupported,dtype=int)


@torch.inference_mode()
def match(net,h,cv,ols,bank,radii):
    """Give the additive controls the joint plan's exact peak slots."""
    arms = bank.shape[1]
    off = bank-cv[:,None]
    peak = np.linalg.norm(off,axis=-1).max(2)
    factor = [np.divide(a,peak,out=np.zeros_like(peak),where=(peak>0.) & (peak>=a-1e-10))
              for a in radii]
    paths = cv[:,None]+np.concatenate([off*np.minimum(f,1.)[...,None,None] for f in factor],axis=1)
    x,_,legal,_ = costs.features(h,cv,ols,paths)
    score = net.unary(net.action(torch.from_numpy(x))).squeeze(-1).numpy()
    r,k = np.nonzero(legal[:,:arms])
    if len(radii)==1:
        best = score[r,k].argmax()
        result = cv.copy()
        result[r[best]] = paths[r[best],k[best]]
        return result,result.copy()
    s,l = np.nonzero(legal[:,arms:])
    value = np.where(r[:,None]!=s[None],score[r,k,None]+score[s,l+arms][None],-np.inf)

    def make(v):
        if not np.isfinite(v).any():
            return None
        i,j = np.unravel_index(int(v.argmax()),v.shape)
        p = cv.copy()
        p[r[i]],p[s[j]] = paths[r[i],k[i]],paths[s[j],l[j]+arms]
        return p

    independent = make(value)
    if independent is None:
        raise ValueError('Joint actions must remain legal in matched slots')
    return independent,make(np.where(k[:,None]==l[None],value,-np.inf))
