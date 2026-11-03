"""Past-defined neighbor geometry diagnostic, never a physical safety certificate."""
import numpy as np


def past_edges(positions, agent_ids, neighbors=8):
    xy, ids = np.asarray(positions), np.asarray(agent_ids)
    if (xy.shape != (len(ids), 2) or ids.ndim != 1 or len(set(ids)) != len(ids)
            or not np.isfinite(xy).all() or type(neighbors) is not int or neighbors < 1):
        raise ValueError('Unique current agents and finite past/current positions required')
    edges = set()
    for i in range(len(ids)):
        others = np.flatnonzero(np.arange(len(ids)) != i)
        distance = np.square(xy[others]-xy[i]).sum(1)
        order = np.lexsort((ids[others], distance))
        for j in others[order[:neighbors]]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return np.asarray(sorted(edges), np.int64).reshape(-1, 2)


def errors(prediction, truth, valid):
    p, y, mask = np.asarray(prediction, float), np.asarray(truth, float), np.asarray(valid)
    if (p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (12, 2)
            or mask.shape != p.shape[:2] or mask.dtype != bool or not np.isfinite(p).all()
            or not np.isfinite(y[mask]).all()):
        raise ValueError('Finite twelve-step predictions and separate supported labels required')
    distance = np.linalg.norm(np.where(mask[..., None], p-y, 0), axis=-1)
    known = mask.any(1)
    ade = np.divide(distance.sum(1), mask.sum(1), out=np.full(len(p), np.nan), where=known)
    fde = np.where(mask[:, -1], distance[:, -1], np.nan)
    return ade, fde


def score(edges, predictions, truth, valid):
    """All positions are displacements from each agent's own observed origin."""
    edges = np.asarray(edges)
    y, mask = np.asarray(truth), np.asarray(valid)
    if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu'
            or np.any(edges < 0) or np.any(edges >= len(y))
            or len({tuple(v) for v in edges}) != len(edges)
            or set(predictions) != {'cv', 'floor', 'pct', 'eqmotion'}):
        raise ValueError('Unique past-defined edges and all four fixed predictors required')
    a, b = edges.T
    labels = y[a]-y[b]
    support = mask[a]&mask[b]
    relative = {key:np.asarray(p)[a]-np.asarray(p)[b] for key,p in predictions.items()}
    known = support.any(1)
    endpoint = support[:, -1]
    metrics = {}
    costs = {}
    for key,p in relative.items():
        ade, fde = errors(p, labels, support)
        costs[key] = ade
        metrics[key] = dict(available_pair_ADE_pixel=float(ade[known].mean()) if known.any() else None,
            available_endpoint_pair_FDE_pixel=float(fde[endpoint].mean()) if endpoint.any() else None)
    denominator = float(costs['cv'][known].sum())
    observed = float((costs['pct'][known]-costs['eqmotion'][known]).sum())
    unknown_bound = float(np.linalg.norm(relative['pct']-relative['eqmotion'], axis=-1).max(1)[~known].sum())
    fraction = lambda n:100*n/denominator if denominator > 0 else None
    return dict(edges=len(edges), known_pairs=int(known.sum()), complete12_pairs=int(support.all(1).sum()),
        wholly_unknown_pairs=int((~known).sum()), endpoint_pairs=int(endpoint.sum()), metrics=metrics,
        known_pair_CV_mass=denominator, unknown_pair_disagreement_bound=unknown_bound,
        paired_known_pp=fraction(observed), paired_completion_lower_pp=fraction(observed-unknown_bound),
        paired_completion_upper_pp=fraction(observed+unknown_bound),
        label_masks_used_for_edge_construction=False, partial_pair_ADE_not_full12_truth=True,
        common_translation_blind=True, physical_safety_or_metric_claim=False)
