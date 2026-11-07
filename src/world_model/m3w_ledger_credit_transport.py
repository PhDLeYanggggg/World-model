"""Audit and discount only inter-query transport of predicted budget credit."""
import numpy as np

from src.world_model import m3w_causal_record_ledger as ledger


def query_totals(groups, choice, a, e):
    zero = np.zeros_like(a)
    return np.stack([ledger.totals(choice[g],zero[g],a[g],e[g])[1:] for g in groups])


def earned_credit(predicted, observed):
    """Actual signed savings on exactly the queries that predict savings."""
    earning = predicted < -ledger.EPS
    return dict(predicted=(-np.where(earning,predicted,0.).sum(0)).tolist(),
        realized=(-np.where(earning,observed,0.).sum(0)).tolist(),
        earning_queries=earning.sum(0).tolist(),
        positive_observed_on_credit=(earning & (observed > ledger.EPS)).sum(0).tolist())


def calibrate(donor_records, held):
    if held in donor_records:
        raise ValueError('Held source cannot determine the credit discount')
    scores = []
    for rows in donor_records.values():
        p = np.sum([r['predicted'] for r in rows],axis=0)
        y = np.sum([r['realized'] for r in rows],axis=0)
        scores.append(np.divide(y,p,out=np.full(2,np.nan),where=p>ledger.EPS).clip(0.,1.))
    scores = np.stack(scores)
    finite = np.isfinite(scores)
    alpha = [float(np.quantile(scores[finite[:,i],i],.10,method='lower')) if finite[:,i].sum()>=3 else 0.
             for i in range(2)]
    return dict(alpha=alpha,available_donors=finite.sum(0).tolist(),
        donor_ratios=[[float(z) if np.isfinite(z) else None for z in row] for row in scores])


def stream(groups, value, a, e, allowed, counts, witness, alpha):
    alpha = np.asarray(alpha,float)
    if alpha.shape!=(2,) or not np.isfinite(alpha).all() or (alpha<0).any() or (alpha>1).any():
        raise ValueError('Two finite credit retentions in [0,1] required')
    slack = np.zeros(2); out = np.full(len(value),-1,dtype=np.int16); borrowed = 0
    for g, n in zip(groups,counts,strict=True):
        u, ac, ec, ok = (z[g] for z in (value,a,e,allowed))
        choice = ledger.query_choice(u,ac,ec,ok,slack,int(n))
        if choice is None:
            choice = witness[g]
        ix = np.flatnonzero(choice>=0)
        cost = ledger.totals(choice,u,ac,ec)[1:]
        if len(ix)!=n or not ok[ix,choice[ix]].all() or (cost>slack+ledger.EPS).any():
            raise ValueError('Invalid matched count or prefix spending')
        slack -= np.maximum(cost,0.) + alpha*np.minimum(cost,0.)
        out[g] = choice; borrowed += int((cost>ledger.EPS).any())
    return out, dict(predicted_final_slack=slack.tolist(),queries_using_credit=borrowed)
