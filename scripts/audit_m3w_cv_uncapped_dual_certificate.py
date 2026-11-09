"""Bound unresolved uncapped cases via a separable per-recipient dual."""
from collections import Counter
from pathlib import Path
import sys

from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_pooled_coverage_capacity as original

study,audit,np,base,prior = original.study,original.audit,original.np,original.base,original.prior
read,sha,artifact,once,stable = study.read,study.sha,study.artifact,study.once,study.stable


def oracle(d):
    cost = np.column_stack((d['risk'],-d['easy'].astype(float)))
    scale = np.abs(cost).max(0).clip(1.)
    cost = cost/scale
    limit = np.array([0.,0.,d['easy_cap'],-1.])/scale
    gain,offset = d['gain'],d['offset']
    row_start = np.r_[0,np.flatnonzero(np.diff(d['row']))+1]
    col = np.arange(len(gain))
    group = np.cumsum(np.r_[True,np.diff(d['row'])!=0])-1
    diff = gain[:,0]-gain[:,1]

    def value(v):
        theta,price = v[0],v[1:]
        score = gain[:,1]+theta*diff-cost@price
        best = np.maximum.reduceat(score,row_start)
        ind = np.minimum.reduceat(np.where(score==best[group],col,len(col)),row_start)[best>0.]
        total = offset+gain[ind].sum(0)
        spend = cost[ind].sum(0)
        upper = offset[1]+theta*(offset[0]-offset[1])+limit@price+best.clip(0.).sum()
        gradient = np.r_[total[0]-total[1],limit-spend]
        feasible = (spend<=limit+1e-8).all() and min(total)>audit.model.POSITIVE_PP
        return float(upper),gradient,ind if feasible else None

    return value


def certify(d,args,cap):
    f = oracle(d)
    bounds = [(0.,1.)]+[(0.,None)]*4
    best,witness = np.inf,None

    def objective(v):
        nonlocal best,witness
        # Explicit projection ensures even solver trial points are dual feasible.
        x = np.r_[v[0].clip(0.,1.),v[1:].clip(0.)]
        val,grad,chosen = f(x)
        best = min(best,val)
        if chosen is not None and witness is None:
            a = np.column_stack((d['row'][chosen],d['arm'][chosen])).tolist()
            r = audit.verify(a,*args,cap)
            if r['verified']:
                witness = dict(actions=a,edits=r['edits'],gain_pp=r['gain_pp'])
        return val,grad

    for theta in (.5,0.,1.):
        minimize(objective,np.r_[theta,0.,0.,0.,0.],jac=True,bounds=bounds,method='L-BFGS-B',
                 options=dict(maxiter=180,ftol=1e-10,maxls=40))
        if witness is not None or best < -audit.model.POSITIVE_PP:
            break
    if witness is not None and best < -audit.model.POSITIVE_PP:
        raise ValueError('Dual bound contradicts a verified witness')
    return dict(status='verified' if witness else 'nonpositive_bound' if best < -audit.model.POSITIVE_PP else 'unresolved',
                upper_pp=best,witness=witness)


def main():
    out,private = study.OUT,study.PRIVATE
    parent = out/'hindsight_capacity.json'
    p = read(parent)
    if p['parent']['sha256']!=sha(out/'readout.json') or p['independent_confirmation'] or p['a_labels_opened']:
        raise ValueError('Require verified initial capacity')
    _,donor,_ = base.register()
    data = base.fit_population(lambda **v:None)
    status = Counter()
    cells = []
    binding = dict(parent=artifact(parent),code=artifact(Path(__file__)))
    for i,c in enumerate(read(out/'freeze.json')['cells']):
        ident = c['identity']
        hi,h = base.subset(data,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
        for rec in np.unique(h['recording']):
            prior_cell = read(private/f'capacity_{i}_{rec}.json')
            if (prior_cell['binding']['parent']!=p['parent'] or prior_cell['binding']['code']!=p['code']
                    or prior_cell['binding']['cell']!=i or prior_cell['binding']['recording']!=int(rec)):
                raise ValueError('Initial capacity checkpoint changed')
            if prior_cell['witness']['uncapped']['status']!='unresolved':
                continue
            ix = h['recording']==rec
            cv0,o = cv[ix],ols[ix]
            bank = base.expanded(cv0,raw[ix])
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            args = (cv0,o,bank,y,v,q,*c['cut'])
            count = np.unique(q,return_counts=True)[1].max()
            path = private/f'dual_{i}_{rec}.json'
            if not path.exists():
                d = audit.model.coefficients(cv0,o,bank,y,v,q,c['cut'][0])
                z = certify(d,args,int(count))
                once(path,stable(dict(binding=binding,result=z)))
            z = read(path)
            if z['binding']!=binding:
                raise ValueError('Dual checkpoint changed')
            r = z['result']
            if r['witness'] is not None:
                if not audit.verify(r['witness']['actions'],*args,int(count))['verified']:
                    raise ValueError('Invalid continuation witness')
            status[r['status']] += 1
            cells.append(r['upper_pp'])
        print(dict(cell=i,continuations=sum(status.values())),flush=True)
    original_count = p['verified_counts']['uncapped']
    if sum(status.values())!=original_count.get('unresolved',0):
        raise ValueError('Expected every initially unresolved uncapped view')
    combined = Counter({k:v for k,v in original_count.items() if k!='unresolved'})
    combined.update(status)
    once(out/'uncapped_continuation.json',stable(dict(**binding,continuation=dict(status),
        final_uncapped=dict(combined),median_upper_pp=float(np.median(cells)),
        original_2pct=True,independent_confirmation=False,hindsight=True)))
    print(combined,flush=True)


if __name__=='__main__':
    main()
