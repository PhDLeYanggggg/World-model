"""Replay exact-count hindsight witnesses; distinguish menu from ranking limits."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as audit
from scripts import run_m3w_cv_record_allocation as study

np, torch, base, prior, rank = study.np, study.torch, study.base, study.prior, study.rank
m = audit.model
FLOORS = ('cv_only','dual')
BUDGETS = ('query_exact','record_exact')


def solve(d, counts, budget, seconds=1.5):
    n = len(d['row'])
    if not counts.sum() or d['ref']<=0. or not d['easy'].any():
        return None,'undefined_support'
    mat,rhs = m.constraints(d,study.CAP)
    col = np.arange(n)
    if budget=='query_exact':
        index,k = d['group'][d['row']],counts
    elif budget=='record_exact':
        index,k = np.zeros(n,dtype=int),np.array([counts.sum()])
    else:
        raise ValueError('Unknown matching budget')
    equal = m.coo_array((np.ones(n),(index,col)),shape=(len(k),n+1)).tocsc()
    c = np.r_[np.zeros(n),-1.]
    lp = m.linprog(c,A_ub=mat,b_ub=rhs,A_eq=equal,b_eq=k,bounds=[(0.,1.)]*n+[(None,None)],method='highs',
                  options={'time_limit':seconds})
    if lp.status==2:
        return None,'risk_infeasible'
    if lp.status==0 and -lp.fun<=m.POSITIVE_PP:
        return None,'nonpositive_bound'
    if lp.status==0 and np.max(np.abs(lp.x[:-1]-np.round(lp.x[:-1])))<=1e-8:
        return lp.x[:-1]>.5,'unresolved'
    z = m.milp(c,integrality=np.r_[np.ones(n),0.],
        bounds=m.Bounds(np.r_[np.zeros(n),-np.inf],np.r_[np.ones(n),np.inf]),
        constraints=[m.LinearConstraint(mat,-np.inf,rhs),m.LinearConstraint(equal,k,k)],
        options={'time_limit':seconds,'mip_rel_gap':0.})
    if z.status==2:
        return None,'risk_infeasible'
    ub = getattr(z,'mip_dual_bound',None)
    if ub is not None and np.isfinite(ub) and -ub<=m.POSITIVE_PP:
        return None,'nonpositive_bound'
    return (z.x[:-1]>.5 if z.x is not None else None),'unresolved'


def witness(design, args, local, counts, floor, budget):
    if floor=='cv_only':
        design = dict(design,gain=design['gain'][:,[0,0]],offset=np.zeros(2))
    chosen,status = solve(design,counts,budget)
    if chosen is not None:
        a = np.column_stack((design['row'][chosen],design['arm'][chosen]))
        receipt = audit.verify(a,*args,study.CAP)
        at = a[:,0]
        count = np.bincount(design['group'][at],minlength=len(counts))
        wanted = count.sum()==local.sum() and (budget=='record_exact' or np.array_equal(count,counts))
        effect = receipt['gain_pp']
        if (wanted and receipt['original_risk']['supported'] and receipt['edits']==receipt['original_risk']['selected']
                and effect is not None and (effect[0] if floor=='cv_only' else min(effect))>m.POSITIVE_PP):
            return dict(status='verified',actions=a.tolist())
    return dict(status=status,actions=None)


def main():
    read,sha = study.read,study.sha
    out,private = study.OUT,study.PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (private/'capacity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source = out/'readout.json'
        f, r = read(out/'freeze.json'),read(source)
        if r['freeze']!=sha(out/'freeze.json') or r['a_labels_opened'] or r['independent_confirmation']:
            raise ValueError('Require complete fitting-only allocation result')
        code = study.artifact(Path(__file__))
        _,donor,_ = base.register()
        data = base.fit_population(lambda **v:None)
        a,b = [read(p/'freeze.json')['cells'] for p in (base.OUT,rank.OUT)]
        counters = {(x,y):Counter() for x in FLOORS for y in BUDGETS}
        transitions = {x:Counter() for x in FLOORS}
        for k,(cell,old,fit) in enumerate(zip(f['cells'],a,b,strict=True)):
            ident = cell['identity']
            if ident!=old['identity'] or ident!=fit['identity']:
                raise ValueError('Parent fold/seed changed')
            ix,d = base.subset(data,ident['held'])
            seed = ident['seed']
            cv,ols,raw = prior.bank(d['history'],{n:v[ix] for n,v in prior.read_donor(donor,seed,data).items()})
            bank = base.expanded(cv,raw)
            h = d['history']
            width = prior.study.motion.features(h[:1],'prefix')[0].shape[1]
            rn = study.screen.load_head(old['fits'][0],width+25,8)
            pn = study.screen.load_head(fit['fits'][1],width+49,2)
            path = rank.nomination(rank.forecast(pn,cv,ols,bank,h),
                study.screen.predictions(rn,cv,bank,h),study.screen.RELAXED)[0]
            fixed = prior.apply(cv,bank,path)[:,None]
            with np.load(study.checked(cell['actions']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['ids'])
                local = z['local']
                np.testing.assert_array_equal(path[local>=0],local[local>=0])
            for record in np.unique(d['recording']):
                at = d['recording']==record
                c,o,p,y,v,q = cv[at],ols[at],fixed[at],d['target'][at],d['valid'][at],d['query'][at]
                cut = cell['cut']
                design = m.coefficients(c,o,p,y,v,q,cut[0])
                selection = local[at]>=0
                counts = np.bincount(design['group'][selection],minlength=len(design['counts']))
                cache = private/f'capacity_{k}_{record}.json'
                binding = dict(parent=study.artifact(source),code=code,cell=k,recording=int(record))
                if not cache.exists():
                    status = {floor:{budget:witness(design,(c,o,p,y,v,q,*cut),selection,counts,floor,budget)
                                     for budget in BUDGETS} for floor in FLOORS}
                    study.once(cache,study.stable(dict(binding=binding,status=status)))
                obj = read(cache)
                if obj['binding']!=binding:
                    raise ValueError('Capacity checkpoint changed')
                for floor in FLOORS:
                    states = []
                    for budget in BUDGETS:
                        w = obj['status'][floor][budget]
                        if w['status']=='verified':
                            r = audit.verify(w['actions'],c,o,p,y,v,q,*cut,study.CAP)
                            g = r['gain_pp']
                            picked = np.asarray(w['actions'])[:,0]
                            count = np.bincount(design['group'][picked],minlength=len(counts))
                            if (r['edits']!=int(selection.sum()) or not r['original_risk']['supported']
                                    or (budget=='query_exact' and not np.array_equal(count,counts))
                                    or (g[0] if floor=='cv_only' else min(g))<=m.POSITIVE_PP):
                                raise ValueError('Invalid saved witness')
                        states.append(w['status'])
                        counters[floor,budget][w['status']] += 1
                    transitions[floor][' -> '.join(states)] += 1
                prior.donor.prior.prior.source.atomic(private/'capacity_heartbeat.json',
                    dict(pid=os.getpid(),cell=k,recording=int(record)))
            print(dict(pid=os.getpid(),cell=k,seed=seed),flush=True)
        if any(sum(x.values())!=72 for x in counters.values()):
            raise ValueError('Expected all 72 recording views')
        result = dict(parent=study.artifact(source),code=code,seconds_per_solve=1.5,
            counts={f'{x}_{y}':dict(v) for (x,y),v in counters.items()},transitions={k:dict(v) for k,v in transitions.items()},
            hindsight=True,original_2pct=True,independent_confirmation=False)
        study.once(out/'hindsight_capacity.json',study.stable(result))
        print(result['counts'],flush=True)


if __name__=='__main__':
    main()
