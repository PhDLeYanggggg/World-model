"""Relax query-local peak limits to a recording-wide intervention budget."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_exchange_peak_envelope as parent

study,old,base,prior,np,model = parent.study,parent.old,parent.base,parent.prior,parent.np,parent.model
FACTORS = parent.FACTORS


def pool(dist,w,peak,factors=FACTORS):
    n,t = dist.shape
    k = int((peak>0.).sum())
    if not k:
        return np.zeros(len(factors))
    d = np.c_[np.zeros(n),np.sort(dist,axis=1)]
    pref = d.cumsum(1)
    cost = np.unique(np.r_[0.,np.geomspace(.0001,float(t),128),np.linspace(0.,t,49)])
    budget = float(peak.sum())
    best = np.full(len(factors),np.inf)
    row = np.arange(n)
    for price in cost:
        x = price/w
        j = np.ceil(np.maximum(0.,t-x)).astype(int)
        profit = w*(pref[row,j]+(t-j-x)*d[row,j])
        top = np.partition(profit,n-k)[n-k:].sum()
        z = np.array([top+price*budget*f if f is not None else np.inf for f in factors])
        if price==0.:
            z[[f is None for f in factors]] = top
        best = np.minimum(best,z)
    return best


def bound(c,o,u,y,v,q):
    y = np.where(v[...,None],y,0.)
    e = np.linalg.norm(c-y,axis=-1)
    f = np.linalg.norm(o-y,axis=-1)
    gap = np.linalg.norm(c-o,axis=-1)
    w = old.weight(q)
    ref = w@np.where(v,e,0.).sum(1)
    if ref<=0.:
        return None
    base = w@np.where(v,f-e,-gap).sum(1)
    delta = pool(np.where(v,e,gap),w,model.peak(u-c))
    return (100.*(base+delta)/ref).tolist()


def main():
    r,f,p = [study.OUT/k for k in ('readout.json','freeze.json','peak_envelope.json')]
    a,b,q = map(old.read,(r,f,p))
    if (a['freeze']!=old.sha(f) or q['parent']['sha256']!=old.sha(r)
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct'] for z in (a,q))):
        raise ValueError('Expected original matched count and peak-envelope result')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    rows = []
    for c in b['cells']:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
        bank = base.expanded(cv,raw)
        with np.load(old.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            u = cv+.5*(old.slate.path_for(cv,bank,z['anchor'])-cv)
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q0 = (h[k][ix] for k in ('target','valid','query'))
            z = bound(cv[ix],ols[ix],u[ix],y,v,q0)
            if z is not None:
                rows.append(z)
        print(dict(fold=ident['fold'],seed=ident['seed'],views=len(rows)),flush=True)
    x = np.asarray(rows)
    if x.shape!=(72,len(FACTORS)) or (np.diff(x,axis=1)<-1e-8).any():
        raise ValueError('Incomplete or nonmonotone recording envelope')
    out = {str(factor):dict(ols4_nonpositive=int((x[:,j]<=0.).sum()),median_upper_pp=float(np.median(x[:,j])))
           for j,factor in enumerate(FACTORS)}
    if any(out[k]['ols4_nonpositive']>q['summary'][k]['matched_count']['ols4_nonpositive'] for k in out):
        raise ValueError('Recording envelope excludes more views than query envelope')
    old.once(study.OUT/'record_peak_envelope.json',old.stable(dict(parent=old.artifact(p),code=old.artifact(Path(__file__)),
        summary=out,views=len(rows),budget_relative_to='half_anchor',ignores_risk=True,
        a_labels_opened=False,independent_confirmation=False,original_2pct=True,labels_hindsight_only=True)))
    print(out,flush=True)


if __name__=='__main__':
    main()
