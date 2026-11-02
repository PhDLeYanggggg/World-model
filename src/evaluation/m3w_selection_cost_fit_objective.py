"""Full decoded fitting objective with the original query-draw multiplicities."""
import numpy as np


def score(prediction, truth, allowed, known, scales, rms, groups, counts, sites):
    p = np.asarray(prediction, float); y = np.asarray(truth, float)
    allowed, known = map(np.asarray,(allowed,known))
    if allowed.dtype != bool or known.dtype != bool or known.shape != (len(p),):
        raise ValueError('Explicit boolean legal-action and known-target masks required')
    legal = allowed & known[:,None]
    scales, rms, counts, sites = map(np.asarray,(scales,rms,counts,sites))
    if (p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (7,5) or legal.shape != p.shape[:2]
            or scales.shape != (len(p),) or rms.shape != (5,) or counts.shape != (len(groups),)
            or sites.shape != (len(p),) or counts.dtype.kind not in 'iu'
            or not np.isfinite(p).all() or not np.isfinite(y[known]).all()
            or not np.isfinite(scales).all() or not np.isfinite(rms).all()
            or (p < 0).any() or (scales <= 0).any() or (rms <= 0).any() or (counts <= 0).any()):
        raise ValueError('Finite decoded costs, observed training labels and original positive weights required')
    if not groups or not np.array_equal(np.sort(np.concatenate(groups)),np.arange(len(p))):
        raise ValueError('Every SOURCE row must belong to exactly one query')
    normalized = p/scales[:,None,None]
    target = np.where(np.asarray(known)[:,None,None],y,0.)
    error = np.square((normalized-target)/rms)
    gain = np.square(((normalized[:,:,0]-normalized[:,:,1])-(target[:,:,0]-target[:,:,1]))/np.linalg.norm(rms[:2]))
    n = legal.sum(1).clip(1)
    per_row = (error*legal[:,:,None]).sum(1)/n[:,None]
    per_gain = (gain*legal).sum(1)/n
    supported = legal.any(1); cells = []
    for q,ix in enumerate(groups):
        if len(set(sites[ix])) != 1:
            raise ValueError('A query cannot mix SOURCE localities')
        use = ix[supported[ix]]
        cost = per_row[use].mean(0) if len(use) else np.zeros(5)
        g = float(per_gain[use].mean()) if len(use) else 0.
        cells.append(dict(site=str(sites[ix[0]]),visits=int(counts[q]),components=cost,
                          gain=g,total=float(cost.mean()+.5*g),observed_agents=len(use)))
    output = {}
    for site in sorted(set(sites)):
        rows = [r for r in cells if r['site'] == site]; total = sum(r['visits'] for r in rows)
        if not total: raise ValueError('All SOURCE localities require query coverage')
        avg = lambda fn:float(sum(r['visits']*fn(r) for r in rows)/total)
        output[site] = dict(total=avg(lambda r:r['total']),gain_MSE=avg(lambda r:r['gain']),
            components={k:avg(lambda r,i=i:r['components'][i]) for i,k in enumerate(('B','H','R','ER','EH'))},
            query_visits=total,queries=len(rows),unlabeled_queries=sum(not r['observed_agents'] for r in rows))
    total = sum(r['query_visits'] for r in output.values())
    return dict(total=sum(r['query_visits']*r['total'] for r in output.values())/total,by_site=output,
                query_visits=total,queries=len(groups),unknown_supervised_rows=0,
                decoded_costs=True,float64_aggregate_not_original_float32_minibatch_trace=True)
