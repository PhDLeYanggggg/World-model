"""Fit gain and risk conditional on a fixed family-diverse shortlist."""
import numpy as np

from src.world_model.m3w_easy_joint_anchor import predict


def features(raw,scale,utility,all_cost,easy_cost,legal,rank,bank,floor,groups):
    joint,marginal = predict(raw,rank)
    family_count = legal.shape[1]
    radius = np.linalg.norm(bank-floor[:,None],axis=-1).max(2)/scale[:,None]
    out = []
    for q in groups:
        if len(q)<2:
            continue
        old = joint[q].ravel()
        valid = np.flatnonzero(legal[q].ravel())
        if not len(valid):
            continue
        flat = valid[np.argsort(-old[valid],kind='stable')]
        seen,first,rest = set(),[],[]
        for z in flat:
            f = z%family_count
            if f not in seen:
                seen.add(f)
                first.append(z)
            else:
                rest.append(z)
        take = np.array((first+rest)[:8])
        i,j = q[take//family_count],take%family_count
        score = joint[i,j]
        x = np.column_stack((np.ones(len(i)),score,marginal[i,j],np.arcsinh(utility[i,j]/scale[i]),
            np.arcsinh(all_cost[i,j]/scale[i]),np.arcsinh(easy_cost[i,j]/scale[i]),
            np.log1p(radius[i,j]),np.log1p(scale[i]),np.full(len(i),np.log1p(len(q))),
            np.arange(len(i))/8,score-joint[i[0],j[0]],np.eye(family_count)[j]))
        out.append((q,i,j,x))
    return out


def targets(atoms,scale,examples):
    y = []
    for _,i,j,_ in examples:
        ref,easy,harm,easy_harm,signed,utility = atoms[i,j].T
        good = (easy>0.) & (harm<=.02*ref) & (utility>0.)
        numeric = np.stack((utility,harm-.02*ref,easy_harm-.02*easy,signed),axis=1)/scale[i,None]
        y.append(np.column_stack((numeric,good)))
    return y


def moments(examples,y,direct):
    if not examples:
        return None
    p = examples[0][3].shape[1]
    g,b,total = np.zeros((p,p)),np.zeros((p,5)),0.
    # The first direct row (if any) is defined in chronological query order.
    prefix = True
    for (q,_,_,x),t in zip(examples,y):
        prefix &= not (direct[q]>=0).any()
        weight = (4. if prefix else 1.)/len(x)
        g += weight*(x.T@x)
        b += weight*(x.T@t)
        total += weight*len(x)
    return g/total,b/total


def fit(moments):
    by_source = [[v for v in s if v is not None] for s in moments]
    by_source = [s for s in by_source if s]
    g,b = [np.mean([np.mean([v[j] for v in s],axis=0) for s in by_source],axis=0) for j in (0,1)]
    ridge = np.eye(g.shape[0])*.02
    ridge[0,0] = 0.
    return np.linalg.solve(g+ridge,b)


def winners(example,coef):
    _,i,j,x = example
    utility,all_cost,easy_cost,signed,p = (x@coef).T
    score = (utility,utility-np.maximum(all_cost,0.)-np.maximum(easy_cost,0.)-np.maximum(signed,0.))
    chosen = [int(s.argmax()) for s in score]
    return [(float(np.clip(p[z],0.,1.)),int(i[z]),int(j[z])) for z in chosen]
