"""Past-only forecast-prefix cues and separate descriptive training-role costs."""
import numpy as np

PREFIXES=(1,2,3,4,8,12)
NAMES=('last_speed','mean_past_speed','last_acceleration','recent_turn_cos','recent_turn_sin')+tuple(
    f'prefix{k}_{name}' for k in PREFIXES for name in
    ('heading_cos','heading_valid','speed_ratio','floor_departure_forward','floor_departure_side'))
BINS=('undefined_direction','opposing','oblique','aligned','near_aligned')


def features(history,forecast,floor):
    h,p,b=map(np.asarray,(history,forecast,floor));n=len(h)
    if (not n or h.shape!=(n,8,2) or p.shape!=(n,12,2) or b.shape!=p.shape
            or not all(np.isfinite(v).all() for v in (h,p,b))):
        raise ValueError('Eight past positions and two finite causal relative rollouts required')
    v=np.diff(h,axis=1);speed=np.linalg.norm(v,axis=-1);last=v[:,-1]
    scale=np.maximum(speed.mean(1),1.)
    direction=np.divide(last,speed[:,-1,None],out=np.zeros_like(last),where=speed[:,-1,None]>0)
    aa,bb=v[:,-2],v[:,-1];den=speed[:,-2]*speed[:,-1]
    cos=np.divide((aa*bb).sum(1),den,out=np.zeros(n),where=den>0)
    sin=np.divide(aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0],den,out=np.zeros(n),where=den>0)
    out=[speed[:,-1],speed.mean(1),np.linalg.norm(bb-aa,axis=1),cos,sin]
    for k in PREFIXES:
        endpoint=p[:,k-1];length=np.linalg.norm(endpoint,axis=1)
        valid=(speed[:,-1]>0)&(length>0)
        angle=np.divide((endpoint*direction).sum(1),length,out=np.zeros(n),where=valid)
        delta=p[:,k-1]-b[:,k-1]
        forward=(delta*direction).sum(1)/(k*scale)
        side=(direction[:,0]*delta[:,1]-direction[:,1]*delta[:,0])/(k*scale)
        out.extend((np.clip(angle,-1,1),valid.astype(float),length/(k*scale),forward,side))
    return np.stack(out,1)


def direction_bin(cosine,valid):
    c,v=np.asarray(cosine),np.asarray(valid)
    if c.shape!=v.shape or v.dtype!=bool or not np.isfinite(c).all():
        raise ValueError('Finite causal direction and explicit validity required')
    return np.where(~v,0,np.where(c<0,1,np.where(c<.5,2,np.where(c<.9,3,4))))


def first_cos_from_existing120(x):
    x=np.asarray(x)
    if x.shape!=(len(x),120) or not np.isfinite(x).all():raise ValueError('Finite original raw120 required')
    # Relation candidate-start delta plus last observed speed recovers forward motion.
    forward=x[:,74].astype(float)+x[:,20].astype(float);side=x[:,75].astype(float)
    length=np.hypot(forward,side);valid=(x[:,20]>1e-12)&(length>1e-12)
    return np.divide(forward,length,out=np.zeros(len(x)),where=valid),valid


def labels(forecast,floor,target,valid):
    p,b,y=map(np.asarray,(forecast,floor,target));m=np.asarray(valid);n=len(p)
    if (p.shape!=(n,12,2) or b.shape!=p.shape or y.shape!=p.shape or m.shape!=(n,12)
            or m.dtype!=bool or not np.isfinite(p).all() or not np.isfinite(b).all()
            or not np.isfinite(y[m]).all()):
        raise ValueError('Aligned finite predictions and separately supplied evaluation labels required')
    pe=np.linalg.norm(np.where(m[...,None],p-y,0),axis=-1).cumsum(1)
    be=np.linalg.norm(np.where(m[...,None],b-y,0),axis=-1).cumsum(1)
    count=m.cumsum(1);model=np.full(count.shape,np.nan);base=np.full(count.shape,np.nan)
    np.divide(pe,count,out=model,where=count>0);np.divide(be,count,out=base,where=count>0)
    at=np.array(PREFIXES)-1
    return dict(model=model[:,at],floor=base[:,at],count=count[:,at],
        known=count[:,at]>0,complete=count[:,at]==np.array(PREFIXES),
        gain=(base-model)[:,at],harm=np.maximum(model-base,0)[:,at],
        envelope=np.maximum.accumulate(np.linalg.norm(p-b,axis=-1),axis=1)[:,at])


def summarize(truth,weights,easy,group):
    w=np.asarray(weights);e=np.asarray(easy);g=np.asarray(group);n=len(w)
    if (w.shape!=(n,) or (w<0).any() or not np.isfinite(w).all() or w.sum()<=0
            or e.shape!=(n,) or e.dtype!=bool or g.shape!=(n,)):
        raise ValueError('Positive causal-query weights and explicit diagnostic class required')
    rows=[]
    for j,k in enumerate(PREFIXES):
        known=truth['known'][:,j];ee=e&known
        denominator=float(w[known]@truth['floor'][known,j])
        gain=float(w[known]@truth['gain'][known,j]);unknown=float(w[~known]@truth['envelope'][~known,j])
        eh=float(w[ee]@truth['harm'][ee,j]);positive=known&(truth['gain'][:,j]>0)
        benefit=float(w[positive]@truth['gain'][positive,j])
        bins=[]
        for code,name in enumerate(BINS):
            at=g==code;ek=at&ee;bk=at&positive;kk=at&known
            bins.append(dict(name=name,rows=int(at.sum()),weight_share=float(w[at].sum()/w.sum()),
                known_rows=int(kk.sum()),known_easy_rows=int(ek.sum()),
                easy_harm_mass_share=(float(w[ek]@truth['harm'][ek,j]/eh) if eh>0 else None),
                positive_gain_mass_share=(float(w[bk]@truth['gain'][bk,j]/benefit) if benefit>0 else None),
                mean_known_harm=(float(w[kk]@truth['harm'][kk,j]/w[kk].sum()) if w[kk].sum()>0 else None)))
        rows.append(dict(prefix=k,rows=n,known_rows=int(known.sum()),wholly_unknown_rows=int((~known).sum()),
            complete_prefix_rows=int(truth['complete'][:,j].sum()),
            partial_prefix_rows=int((known&~truth['complete'][:,j]).sum()),
            known_easy_rows=int(ee.sum()),known_gain_percent=(100*gain/denominator if denominator>0 else None),
            completion_lower_percent=(100*(gain-unknown)/denominator if denominator>0 else None),
            easy_harm_mass=eh,positive_gain_mass=benefit,direction_bins=bins))
    return rows
