"""Training-label geometry oracle; never a forecast or deployment input."""
import numpy as np


def project_triangle(target, cap, weight):
    """Weighted Euclidean projection onto 0 <= easy <= harm <= cap."""
    t, cap, w = np.asarray(target, float), np.asarray(cap, float), np.asarray(weight, float)
    if (t.ndim != 2 or t.shape[1] != 2 or cap.shape != (len(t),) or w.shape != t.shape
            or not all(np.isfinite(v).all() for v in (t, cap, w)) or (cap < 0).any() or (w <= 0).any()):
        raise ValueError('Finite two-channel targets, nonnegative capacities and positive weights required')
    projected = np.clip(t, 0, cap[:, None])
    tied = projected[:, 1] > projected[:, 0]
    # Use original targets, not their clipped values, for the active shared edge.
    value = np.clip((t*w).sum(1)/w.sum(1), 0, cap)
    projected[tied] = value[tied, None]
    return projected


def diagnose(prediction, envelope, targets, scales, allowed, initial, groups, rms):
    p, env, y, s, rms = map(lambda v: np.asarray(v, float), (prediction, envelope, targets, scales, rms))
    initial, allowed = np.asarray(initial), np.asarray(allowed)
    n = len(p)
    if (p.shape != (n, 7, 5) or y.shape != p.shape or env.shape != (n, 7)
            or s.shape != (n,) or allowed.shape != env.shape or allowed.dtype != bool
            or initial.shape != (n,) or not np.issubdtype(initial.dtype, np.integer)
            or ((initial < -1) | (initial > 6)).any() or rms.shape != (5,)
            or not all(np.isfinite(v).all() for v in (p, env, s, rms))
            or (p < 0).any() or (env < 0).any() or (s <= 0).any() or (rms <= 0).any()):
        raise ValueError('Aligned frozen predictions, causal scales and separate label costs required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any():
        raise ValueError('Unknown five-moment labels must remain entirely unknown')
    if not groups or sorted(np.concatenate(groups).tolist()) != list(range(n)):
        raise ValueError('An exhaustive disjoint original query partition is required')
    at = np.flatnonzero(initial >= 0)
    if not allowed[at, initial[at]].all(): raise ValueError('Only legal original selected actions')
    real = allowed & known[:, None]
    capacity = np.maximum(np.maximum(env-p[:, :, 0], 0), p[:, :, 1])
    normalized_cap = (capacity/s[:, None])[real]
    target = (y[:, :, [1, 4]]/s[:, None, None])[real]
    weights = np.broadcast_to(1/rms[[1, 4]]**2, target.shape)
    point_projected = project_triangle(target, normalized_cap, weights)
    point_error = ((target-point_projected)**2*weights).mean(1)
    result = dict(queries=len(groups), complete_selected_queries=0, unknown_selected_queries=0,
        empty_selected_queries=0, known_legal_actions=int(real.sum()),
        point_harm_above_protected_cap_actions=int((target[:, 0] > normalized_cap+1e-10).sum()),
        point_easy_harm_above_protected_cap_actions=int((target[:, 1] > normalized_cap+1e-10).sum()),
        point_projection_lower_MSE_sum=float(point_error.sum()),
        query_projection_lower_MSE_sum=0., query_target_squared_sum=[0., 0.],
        query_target_below_zero_harm=0, query_target_below_zero_easy_harm=0,
        query_target_above_all_harm_cap=0, query_target_above_easy_harm_cap=0,
        query_target_easy_exceeds_all_harm=0, label_aware_diagnostic_only=True,
        new_policy_scored=False, new_model_trained=False, conditional_mean_misspecification_proved=False)
    squares = np.zeros(2); score_rms = np.sqrt(rms[[1, 4]]**2+(.02*rms[[2, 3]])**2)
    for g in groups:
        ix = g[initial[g] >= 0]
        if not len(ix): result['empty_selected_queries'] += 1; continue
        if not known[ix].all(): result['unknown_selected_queries'] += 1; continue
        c = initial[ix]; truth, pred = y[ix, c], p[ix, c]; mass = s[ix].sum()
        signed = (truth[:, [1, 4]]-.02*truth[:, [2, 3]]).sum(0)/mass
        desired = signed+.02*pred[:, [2, 3]].sum(0)/mass
        cap = capacity[ix, c].sum()/mass
        projected = project_triangle(desired[None], np.array([cap]), (1/score_rms**2)[None])[0]
        result['query_projection_lower_MSE_sum'] += float(np.mean(((desired-projected)/score_rms)**2))
        result['complete_selected_queries'] += 1; squares += signed**2
        for key, condition in (('query_target_below_zero_harm', desired[0] < -1e-10),
                ('query_target_below_zero_easy_harm', desired[1] < -1e-10),
                ('query_target_above_all_harm_cap', desired[0] > cap+1e-10),
                ('query_target_above_easy_harm_cap', desired[1] > cap+1e-10),
                ('query_target_easy_exceeds_all_harm', desired[1] > desired[0]+1e-10)):
            result[key] += int(condition)
    result['query_target_squared_sum'] = squares.tolist()
    result['current_query_score_RMS'] = score_rms.tolist()
    result['label_query_score_RMS_before_floor'] = (np.sqrt(squares/result['complete_selected_queries']).tolist()
        if result['complete_selected_queries'] else None)
    return result
