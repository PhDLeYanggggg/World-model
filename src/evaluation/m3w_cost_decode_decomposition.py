"""Frozen forest cost-space stages and independently weighted fitting error."""
import numpy as np

from src.world_model import m3w_selection_cost_forest as forest

STAGES = ('unconstrained','shared_reference','projected')


def prediction_stages(state,causal,chunk_size=1024):
    pr = state['preprocess']
    z,env,context = forest.encoded(causal,pr,state['arm'])
    if not isinstance(chunk_size,int) or chunk_size < 1:
        raise ValueError('Positive inference chunk size required')
    model = state['model']; model.set_params(n_jobs=1)
    result = {s:np.empty((len(z),7,5),np.float64) for s in STAGES}
    augmented = np.empty((len(z),7,6),np.float64)
    for lo in range(0,len(z),chunk_size):
        hi = min(len(z),lo+chunk_size)
        rows = np.repeat(np.arange(lo,hi),7); actions = np.tile(np.arange(7),hi-lo)
        encoded = model.predict(forest.feature_rows(z,env,context,rows,actions))
        augmented[lo:hi] = encoded.reshape(-1,7,6)
        raw = encoded/forest.FACTORS
        cost = raw[:,:5]*pr['rms']
        np.testing.assert_allclose(raw[:,5],(cost[:,0]-cost[:,1])/np.linalg.norm(pr['rms'][:2]),rtol=1e-5,atol=2e-6)
        cost = cost.reshape(-1,7,5)
        scale = causal['scales'][lo:hi,None,None]
        result['unconstrained'][lo:hi] = cost*scale
        cost[:,:,2:4] = cost[:,:,2:4].mean(1,keepdims=True)
        result['shared_reference'][lo:hi] = cost*scale
        cost = forest.project_moments(cost.reshape(-1,5),env[lo:hi].ravel()).reshape(-1,7,5)
        result['projected'][lo:hi] = cost*scale
    if not all(np.isfinite(p).all() and (p >= 0).all() for p in result.values()):
        raise FloatingPointError('Finite nonnegative cost-space predictions required')
    return result,augmented


def augmented_objective(prediction,truth,allowed,known,rms,groups,counts):
    p,y,allowed,known,rms,counts = map(np.asarray,(prediction,truth,allowed,known,rms,counts))
    if (p.ndim != 3 or p.shape[1:] != (7,6) or y.shape != (*p.shape[:2],5)
            or allowed.shape != p.shape[:2] or allowed.dtype != bool or known.shape != (len(p),)
            or known.dtype != bool or rms.shape != (5,) or not np.isfinite(rms).all() or (rms <= 0).any()
            or not np.isfinite(p).all() or counts.shape != (len(groups),) or counts.dtype.kind not in 'iu'
            or (counts <= 0).any() or not groups
            or not np.array_equal(np.sort(np.concatenate(groups)),np.arange(len(p)))):
        raise ValueError('Original aligned SOURCE predictions,observations and complete query weights required')
    legal = allowed&known[:,None]; per_agent = legal.sum(1)
    row_weight = np.zeros(len(p),np.float64); zero_visits = 0
    for q,ix in enumerate(groups):
        support = per_agent[ix] > 0
        if support.any():
            take = ix[support]
            row_weight[take] = counts[q]/support.sum()/per_agent[take]
        else:
            zero_visits += int(counts[q])
    rows,actions = np.nonzero(legal)
    target = forest.augmented_targets(y[rows,actions],rms)
    weight = row_weight[rows]
    error = np.square(p[rows,actions]-target).mean(1)
    value = float(np.dot(weight,error)/counts.sum())
    normalized_weight = weight/weight.mean() if len(weight) else weight
    return dict(total=value,all_query_visits=int(counts.sum()),zero_supervision_query_visits=zero_visits,
        supervised_action_views=len(rows),unknown_supervised_rows=0,
        matrix_hashes={k:forest.fingerprint(v) for k,v in dict(targets=target,weights=normalized_weight,
            rows=rows,actions=actions,query_counts=counts).items()},
        target_arithmetic='actual_float32_augmented_forest_targets',
        raw_cost_objective_gap_can_include_float32_target_rounding=True)
