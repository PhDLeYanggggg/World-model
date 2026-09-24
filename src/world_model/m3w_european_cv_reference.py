"""CV-relative repair on unchanged forecasts, rows and source boundaries."""
import numpy as np

from src.world_model.m3w_european_source_intervention import controls

CONTROL_ARMS = ('independent', 'scene_uniform', 'joint', 'unary_exact', 'joint_exact')


def cv_reference_design(design):
    if ('baseline_index' not in design or not len(design['train_ids'])
            or not len(design['held_ids'])
            or np.intersect1d(design['train_ids'], design['held_ids']).size):
        raise ValueError('Disjoint frozen fitting and held rows required')
    # This changes the decision reference, never the saved neural producer.
    return dict(design, baseline_index=1)


def query_decisions(reg, *, history, current_xy, widths, recordings, frames,
                    sites, baseline, candidate, costs, held_ids, query_mask,
                    cost_scale, heartbeat):
    held = np.asarray(held_ids)
    qmask = np.asarray(query_mask)
    ids = held[qmask[held]]
    costs = np.asarray(costs, float)
    if (costs.shape != (len(held), 2) or not np.isfinite(costs).all()
            or np.any(costs < 0) or not np.isfinite(cost_scale) or cost_scale <= 0):
        raise ValueError('Finite nonnegative predicted costs and fitting-only scale required')
    moving = np.linalg.norm(np.diff(history[held], axis=1), axis=2).sum(1) > 0
    pointwise = (costs[:, 0] > costs[:, 1]) & moving
    bits = {arm: np.zeros(len(ids), bool) for arm in CONTROL_ARMS}
    matched, nonzero = np.zeros(len(ids), bool), np.zeros(len(ids), bool)
    local = costs[qmask[held]] / cost_scale
    local_moving = moving[qmask[held]]
    keys = np.column_stack((recordings[ids], frames[ids]))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    summaries = []
    for qi, key in enumerate(unique):
        loc = np.flatnonzero(inverse == qi)
        ix = ids[loc]
        result = controls(local[loc], local_moving[loc], current_xy[ix], widths[ix],
            baseline[ix] + current_xy[ix, None], candidate[ix].astype(float) + current_xy[ix, None],
            budget=reg['predicted_positive_harm_budget'], pair_weight=reg['pair_weight'],
            radius_widths=reg['edge_radius_bbox_widths'], threshold_widths=reg['proximity_threshold_bbox_widths'],
            seconds=reg['solver_seconds'])
        matched[loc], nonzero[loc] = result['matched'], result['matched_nonzero']
        for arm in CONTROL_ARMS:
            bits[arm][loc] = result[arm]['switch']
        if result['matched']:
            counts = [int(bits[k][loc].sum()) for k in ('independent', 'unary_exact', 'joint_exact')]
            if len(set(counts)) != 1:
                raise ValueError('Actual matched decision counts differ')
        row = dict(recording=int(key[0]), frame=int(key[1]), site=str(sites[ix[0]]),
            agents=len(ix), edges=result['edges'], matched=result['matched'],
            matched_nonzero=result['matched_nonzero'], reference_count=result['reference_count'], arms={})
        for arm in CONTROL_ARMS:
            row['arms'][arm] = {k: result[arm][k] for k in ('reason', 'solver_optimal',
                'predicted_constraints_satisfied', 'mean_pair_proxy', 'mean_predicted_gain',
                'mean_predicted_harm', 'switch_rate')}
        summaries.append(row)
        if qi % 96 == 0:
            heartbeat(state='query_controls', query=qi+1, total=len(unique))
    return dict(ids=ids, pointwise_ids=held, pointwise=pointwise, matched=matched,
                matched_nonzero=nonzero, **bits), summaries


def choose_errors(bits, candidate, reference):
    bits, candidate, reference = np.asarray(bits), np.asarray(candidate), np.asarray(reference)
    if (bits.dtype != bool or bits.ndim != 1 or bits.shape != candidate.shape
            or bits.shape != reference.shape
            or not np.array_equal(np.isnan(candidate), np.isnan(reference))):
        raise ValueError('Binary causal choices and identically masked paired costs required')
    return np.where(bits, candidate, reference)
