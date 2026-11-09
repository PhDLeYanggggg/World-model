"""A lower bound on edit coverage needed to beat OLS4, even with perfect paths."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_lead_mixture as study
from scripts import audit_m3w_cv_record_allocation_capacity as cap

np, prior, base = study.np,study.prior,study.base


def minimum(cv,ols,y,valid,query,query_cap):
    clean = np.where(valid[...,None],y,0.)
    c,o = [np.linalg.norm(p-clean,axis=-1) for p in (cv,ols)]
    observed = np.where(valid,c,0.).sum(1)
    missing = np.where(valid,0.,np.linalg.norm(cv-ols,axis=-1)).sum(1)
    g = prior.query_groups(query)
    _,inv,n = np.unique(query,return_inverse=True,return_counts=True)
    w = 1./n[inv]
    ref = float(w@observed)
    if ref<=0.:
        return None
    required = cap.m.POSITIVE_PP*ref/100.
    offset = float(w@(np.where(valid,o-c,0.).sum(1)-missing))
    costs = (w*observed,w*(observed+missing))
    limits = []
    for a,b in zip(costs,(0.,offset),strict=True):
        gains = np.concatenate([np.sort(a[ix])[::-1][:query_cap] for ix in g])
        upper = b+np.r_[0.,np.cumsum(np.sort(gains)[::-1])]
        first = int(np.searchsorted(upper,required,side='right'))
        limits.append(first if first<len(upper) else None)
    return max(limits) if all(k is not None for k in limits) else None


def main():
    read,sha = study.read,study.sha
    out = study.OUT
    source = out/'hindsight_capacity.json'
    parent = read(source)
    if (parent['parent']['sha256']!=sha(out/'readout.json') or parent['a_labels_opened']
            or parent['independent_confirmation']):
        raise ValueError('Require complete fitting-source capacity')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    data = {'cap8':[],'uncapped':[]}
    for c in read(out/'freeze.json')['cells']:
        ident = c['identity']
        _,h = base.subset(d,ident['held'])
        cv,ols = prior.study.motion.bank(h['history'])[:,:2].swapaxes(0,1)
        with np.load(study.checked(c['allocation']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            active = z['local']>=0
        for r in np.unique(h['recording']):
            at = h['recording']==r
            y,v,q = (h[k][at] for k in ('target','valid','query'))
            args = (cv[at],ols[at],y,v,q)
            for name,k in (('cap8',base.CAP),('uncapped',None)):
                data[name].append((int(active[at].sum()),minimum(*args,k)))
    stats = {}
    for name,values in data.items():
        assert len(values)==72
        known = [(a,b) for a,b in values if b is not None]
        stats[name] = dict(views=len(values),above_current=sum(b>a for a,b in known),
            unreachable_or_no_reference=len(values)-len(known),at_or_below_current=sum(b<=a for a,b in known),
            median_current_count=float(np.median([a for a,_ in values])),
            median_required_count=float(np.median([b for _,b in known])),
            median_extra_edits=float(np.median([max(b-a,0) for a,b in known])))
    if stats['cap8']['at_or_below_current']!=parent['perfect_bound']['free_pp']['positive_dual']:
        raise ValueError('Minimum-count and fixed-count bounds disagree')
    study.once(out/'minimum_edits.json',study.stable(dict(parent=study.artifact(source),
        code=study.artifact(Path(__file__)),stats=stats,ignores_risk=True,independent_confirmation=False,
        hindsight=True,original_2pct=True)))
    print(stats)


if __name__=='__main__':
    main()
