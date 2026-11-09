"""Ray-wise amplitude response and causal risk-priced query portfolios."""
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors


AMPLITUDES = (.05,.15,.35,.65,1.)
PRICES = (0.,1.,4.,16.,64.)
M = len(AMPLITUDES)
WIDTH = 5*M+3


def response(z,shape):
    v = z[...,:5*M].reshape(*z.shape[:-1],M,5)
    if shape:
        first = torch.cat((v[...,:1,:2],F.softplus(v[...,:1,2:4]-4.),
                           v[...,:1,4:5]),dim=-1)
        inc = F.softplus(v[...,1:,:]-4.)
        sign = z.new_tensor([-1.,-1.,1.,1.,1.])
        slopes = torch.cat((first,first+(inc*sign).cumsum(-2)),dim=-2)
        dt = z.new_tensor(np.diff((0.,*AMPLITUDES))).reshape(M,1)
        v = (slopes*dt).cumsum(-2)
    else:
        v = torch.cat((v[...,:2],F.softplus(v[...,2:4]-4.),v[...,4:5]),dim=-1)
    m = F.softplus(z[...,5*M:5*M+2]-2.)
    return v,m,z[...,-1]


def targets(c,o,ray,y,v,s,cut):
    p = c[:,None]+np.asarray(AMPLITUDES)[None,:,None,None]*(ray-c)[:,None]
    _,_,g,_ = candidate_terms(c,o,p,y,v)
    cv,_ = errors(c,y,v)
    easy = v.any(1)&(cv>0.)&(cv<=cut)
    mass = np.c_[np.where(v.any(1),cv,0.),np.where(easy,cv,0.)]
    harm = risk_excess(c,p,y,v,cut)+.02*mass[:,None]
    signed = np.stack([np.where(easy,errors(x,y,v)[0]-cv,0.) for x in p.swapaxes(0,1)],axis=-1)
    result = np.concatenate((g/12.,harm,signed[...,None]),axis=-1)/s[:,None,None]
    return result.astype(np.float32),(mass/s[:,None]).astype(np.float32),easy.astype(np.float32)


def loss(net,x,y,m,e,shape):
    p,ref,support = response(net(x),shape)
    return F.smooth_l1_loss(p,y)+.25*F.smooth_l1_loss(ref,m)+.10*F.binary_cross_entropy_with_logits(support,e)


def decode(v,m,legal,s):
    g = (v[...,:2]*s[:,None,None,None]).reshape(len(v),-1,2)
    harm = v[...,2:4]-.02*m[:,:,None,:]
    r = (np.concatenate((harm,v[...,4:5]),axis=-1)*s[:,None,None,None]).reshape(len(v),-1,3)
    l = np.repeat(legal,M,axis=1)
    mass = (m*s[:,None,None]).mean(1)
    return g,r,l,mass


def independent(g,r,l,groups,fraction,match=None):
    score = np.where(l,g.min(-1)-2.*r.clip(0.).sum(-1),-np.inf)
    arm = score.argmax(1)
    val = score[np.arange(len(score)),arm]
    out = np.full(len(g),-1,np.int16)
    for q in groups:
        k = int((match[q]>=0).sum()) if match is not None else max(1,int(len(q)*fraction))
        at = q[np.argsort(-val[q],kind='stable')[:k]]
        at = at[np.isfinite(val[at])]
        if match is None:
            at = at[val[at]>0.]
        elif len(at)!=k:
            raise ValueError('Independent control failed query count match')
        out[at] = arm[at]
    return out


def portfolio(g,r,l,mass,groups,fraction,slack):
    out = np.full(len(g),-1,np.int16)
    for q in groups:
        k = max(1,int(len(q)*fraction))
        target = np.r_[slack*mass[q].sum(0),.02*mass[q,1].sum()]
        gg,rr,ll = g[q],r[q],l[q]
        best = 0.
        for price in PRICES:
            # Summed risk is checked on a prefix after pricing signed terms.
            rank = np.where(ll,gg.mean(-1)-price*(rr[...,0]+2.*rr[...,1]+rr[...,2]),-np.inf)
            arm = rank.argmax(1)
            val = rank[np.arange(len(q)),arm]
            at = np.argsort(-val,kind='stable')[:k]
            at = at[np.isfinite(val[at])]
            a = arm[at]
            if not len(at):
                continue
            cumulative = gg[at,a].cumsum(0).min(1)
            ok = (rr[at,a].cumsum(0)<=target).all(1)
            selected = np.where(ok,cumulative,-np.inf)
            i = int(selected.argmax())
            if selected[i]>best:
                best = selected[i]
                out[q] = -1
                out[q[at[:i+1]]] = a[:i+1]
    return out


def apply(cv,raw,a):
    z = cv.copy()
    at = np.flatnonzero(a>=0)
    ray,step = np.divmod(a[at],M)
    alpha = np.asarray(AMPLITUDES)[step]
    z[at] = cv[at]+alpha[:,None,None]*(raw[at,ray]-cv[at])
    np.testing.assert_array_equal((z!=cv).any((1,2)),a>=0)
    return z


def uniform(cv,raw,a,groups):
    at = np.flatnonzero(a>=0)
    ray,step = np.divmod(a[at],M)
    base = cv.copy()
    base[at] = raw[at,ray]
    radius = np.linalg.norm(base-cv,axis=-1).max(1)
    alpha = np.zeros(len(a))
    alpha[at] = np.asarray(AMPLITUDES)[step]
    for q in groups:
        if radius[q].sum()>0.:
            alpha[q] = (radius[q]@alpha[q])/radius[q].sum()
    return cv+alpha[:,None,None]*(base-cv)
