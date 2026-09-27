"""Same-query expected-utility allocation with frozen signed-risk predictions."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

POLICIES = ('floor', 'independent', 'control_matched_count', 'utility_topk',
    'query_uniform', 'joint_utility')


def allocate(utility, excess, eligible, anchor, ids, *, node_limit=256):
    u, q = np.asarray(utility, float), np.asarray(excess, float)
    e, a, ids = map(np.asarray, (eligible, anchor, ids))
    n=len(u)
    if (u.shape!=(n,) or q.shape!=(n,2) or not np.isfinite(u).all() or not np.isfinite(q).all()
            or any(x.shape!=(n,) or x.dtype!=bool for x in (e,a)) or ids.shape!=(n,)
            or len(np.unique(ids))!=n or (a & ~e).any() or (u[e]<=0).any()
            or not isinstance(node_limit,int) or node_limit<=0):
        raise ValueError('Causal aligned utility, signed-risk and frozen anchor required')
    scale=np.maximum(np.max(np.abs(q),axis=0),1e-12) if n else np.ones(2)
    qs=q/scale
    feasible=lambda bits: bool(np.all(qs[bits].sum(0)<=1e-10))
    if not feasible(a):
        raise ValueError('Frozen anchor violates predicted query budget')
    k=int(a.sum()); top=np.zeros(n,bool)
    order=np.lexsort((ids,-u)); order=order[e[order]]; top[order[:k]]=True
    uniform=e.copy() if e.all() and feasible(e) else np.zeros(n,bool)
    info=dict(rows=n,eligible=int(e.sum()),count=k,solver_called=False,optimal=True,
        fallback=False,reason='unconstrained_topk_feasible')
    if feasible(top):
        joint=top.copy()
    else:
        info['solver_called']=True
        active=np.flatnonzero(e); uu=u[active]; qq=qs[active]
        uscale=max(float(np.max(np.abs(uu))),1e-12)
        matrix=np.vstack((np.ones(len(active)),qq.T))
        result=milp(c=-uu/uscale,integrality=np.ones(len(active)),
            bounds=Bounds(np.zeros(len(active)),np.ones(len(active))),
            constraints=LinearConstraint(matrix,np.array([k,-np.inf,-np.inf]),np.array([k,0.,0.])),
            options={'node_limit':node_limit,'mip_rel_gap':0.})
        candidate=np.zeros(n,bool)
        if result.success and result.x is not None:
            candidate[active]=np.asarray(result.x)>.5
        direct=-float(u@candidate)/uscale
        primal=float(result.fun) if result.fun is not None else np.nan
        dual=float(getattr(result,'mip_dual_bound',np.nan))
        tol=1e-8*(1+abs(direct))
        ok=(result.success and result.x is not None and np.isfinite([primal,dual]).all()
            and np.allclose(result.x,candidate[active],atol=1e-7,rtol=0)
            and int(candidate.sum())==k and not (candidate & ~e).any() and feasible(candidate)
            and abs(direct-primal)<=tol and -tol<=direct-dual<=tol
            and float(u@candidate)>=float(u@a)-1e-10*uscale)
        if ok:
            joint=candidate
            info.update(reason='checked_milp_optimum',node_count=int(getattr(result,'mip_node_count',0)))
        else:
            joint=a.copy()
            info.update(reason='solver_or_numerical_check_failed_anchor',optimal=False,fallback=True)
    if float(u@joint)<=float(u@a):
        joint=a.copy()
    assert int(joint.sum())==k and feasible(joint) and not (joint & ~e).any()
    info.update(changed_agents=int((joint!=a).sum()),topk_changed_agents=int((top!=a).sum()),
        expected_utility_gain=float(u@joint-u@a),predicted_feasible=feasible(joint),
        uniform_selected=int(uniform.sum()))
    return dict(utility_topk=top,query_uniform=uniform,joint_utility=joint),info


def grouped(utility, excess, eligible, anchor, recordings, frames, ids, *, node_limit=256):
    records, frames, ids = map(np.asarray,(recordings,frames,ids))
    n=len(ids)
    if any(x.shape!=(n,) for x in (records,frames)) or len(np.unique(ids))!=n:
        raise ValueError('Unique current-query row identities required')
    groups={}
    for i,key in enumerate(zip(records.tolist(),frames.tolist())):
        groups.setdefault(key,[]).append(i)
    outputs={k:np.zeros(n,bool) for k in ('utility_topk','query_uniform','joint_utility')}
    counters=dict(queries=0,milp_queries=0,solver_fallback_queries=0,changed_queries=0,
        changed_agents=0,expected_utility_gain=0.,uniform_queries=0)
    for positions in groups.values():
        at=np.asarray(positions)
        out,info=allocate(utility[at],excess[at],eligible[at],anchor[at],ids[at],node_limit=node_limit)
        for key,bits in out.items(): outputs[key][at]=bits
        counters['queries']+=1; counters['milp_queries']+=info['solver_called']
        counters['solver_fallback_queries']+=info['fallback']
        counters['changed_queries']+=info['changed_agents']>0
        counters['changed_agents']+=info['changed_agents']
        counters['expected_utility_gain']+=info['expected_utility_gain']
        counters['uniform_queries']+=info['uniform_selected']>0
    return outputs,counters
