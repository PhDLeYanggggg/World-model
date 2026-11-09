"""Bound any reallocation at the same query count and peak mass."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_matched_exchange as study

old,base,prior,np,model = study.old,study.base,study.prior,study.np,study.model
FACTORS = (.5,1.,2.,4.,8.,None)


def query_upper(dist,peaks,count,factors=FACTORS):
    """Per-query Lagrangian bound on sum of radius-limited improvements."""
    n,t = dist.shape
    # For integer price k, max_r(sum_t min(r,d_t)-k*r)
    # equals the sum of the shortest t-k step distances.
    price = np.arange(t+1)
    pay = np.c_[np.zeros(n),np.sort(dist,axis=1).cumsum(1)][:,::-1]
    loose = pay.sum(0)
    tight = np.partition(pay,n-count,axis=0)[n-count:].sum(0) if count else np.zeros(t+1)
    result = []
    for f in factors:
        budget = float(peaks.sum())*f if f is not None else None
        result.append([tight[0],loose[0]] if budget is None else
            [min(tight+price*budget),min(loose+price*budget)])
    return np.asarray(result)/n


def bound(c,o,u,y,v,q):
    y = np.where(v[...,None],y,0.)
    e = np.linalg.norm(c-y,axis=-1)
    f = np.linalg.norm(o-y,axis=-1)
    gap = np.linalg.norm(o-c,axis=-1)
    w = old.weight(q)
    ref = w@np.where(v,e,0.).sum(1)
    if ref<=0.:
        return None
    base = w@np.where(v,f-e,-gap).sum(1)
    dist = np.where(v,e,gap)
    r = model.peak(u-c)
    gain = sum(query_upper(dist[g],r[g],int((r[g]>0.).sum())) for g in prior.query_groups(q))
    return (100.*(base+gain)/ref).tolist()


def main():
    r,f = [study.OUT/k for k in ('readout.json','freeze.json')]
    readout,freeze = map(old.read,(r,f))
    if (readout['freeze']!=old.sha(f) or readout['a_labels_opened']
            or readout['independent_confirmation'] or not readout['original_2pct']):
        raise ValueError('Need verified development-only exchange roster')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    rows = []
    for c in freeze['cells']:
        i = c['identity']
        hi,h = base.subset(d,i['held'])
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,i['seed'],d).items()})
        b = base.expanded(cv,raw)
        with np.load(old.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            u = cv+.5*(old.slate.path_for(cv,b,z['anchor'])-cv)
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            z = bound(cv[ix],ols[ix],u[ix],y,v,q)
            if z is not None:
                rows.append(z)
    a = np.asarray(rows)
    if a.shape!=(72,len(FACTORS),2):
        raise ValueError('Incomplete development envelope')
    if (np.diff(a,axis=1)<-1e-9).any() or (a[:,:,0]>a[:,:,1]+1e-9).any():
        raise ValueError('Relaxed envelope contradicted a restricted one')
    x = {str(f):{k:dict(ols4_nonpositive=int((a[:,i,j]<=0.).sum()),median_upper_pp=float(np.median(a[:,i,j])))
                 for j,k in enumerate(('matched_count','unrestricted_count'))} for i,f in enumerate(FACTORS)}
    old.once(study.OUT/'peak_envelope.json',old.stable(dict(parent=old.artifact(r),code=old.artifact(Path(__file__)),
        views=len(a),budget_relative_to='half_anchor',summary=x,ignores_risk=True,a_labels_opened=False,
        independent_confirmation=False,original_2pct=True,labels_hindsight_only=True)))
    print(x,flush=True)


if __name__=='__main__':
    main()
