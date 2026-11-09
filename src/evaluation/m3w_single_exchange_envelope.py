"""Utility-only upper bounds at fixed chronological and peak budgets."""
import numpy as np

from src.evaluation.m3w_opendd_location import paired


def bound(cv, refs, anchor, y, valid, groups, weight):
    """Relax direction, eligibility and risk; retain one same-query swap."""
    refs = np.asarray(refs)
    base = (paired(refs,anchor,y,valid)[1].sum(-1)*weight).sum(1)
    radius = np.linalg.norm(anchor-cv,axis=-1).max(1)
    takeout = (paired(refs,cv,y,valid)[1]-paired(refs,anchor,y,valid)[1]).sum(-1)*weight
    cap = np.where(valid[None],np.linalg.norm(cv-y,axis=-1)[None],np.linalg.norm(refs-cv,axis=-1))
    exchange = base.copy()
    query = base.copy()
    for g in groups:
        if not np.all(weight[g]==weight[g[0]]):
            raise ValueError('Query weights must be constant within the query')
        if not radius[g].any():
            continue
        # For the broader relaxation each changed path pays no more than
        # its old plus new radius per step, summed across the query.
        query = np.maximum(query,base+2.*valid.shape[1]*weight[g[0]]*radius[g].sum())
        active = g[radius[g]>0.]
        inactive = g[radius[g]==0.]
        for i in active:
            if len(inactive):
                gain = np.minimum(cap[:,inactive],radius[i]).sum(2)*weight[inactive]
                exchange = np.maximum(exchange,base+takeout[:,i]+gain.max(1))
    return base,exchange,query
