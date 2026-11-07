"""Exact free-count headroom for already frozen mutually exclusive paths."""
import numpy as np

from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_selection_forecast_readout import policy


def solve(floor, proposals, target, valid, weights, recording, easy_cut, hard_cut):
    entries = [moments(floor,path,target,valid,easy_cut) for path in proposals]
    g = np.stack([m['diff'] for m in entries],1)
    legal = np.stack([m['radius']>0 for m in entries],1)
    known, easy = entries[0]['known'], entries[0]['easy']
    scores = np.where(legal & known[:,None] & (g>=0),g,-np.inf)
    row = np.arange(len(floor)); best = scores.argmax(1)
    choice = np.where(np.isfinite(scores[row,best]),best,-1)
    if not ((choice>=0)&easy).any():
        excess = np.stack([m['easy_excess'] for m in entries],1)
        safe = legal & easy[:,None] & (excess<=0)
        if not safe.any():
            return dict(feasible=False, reason='no_positive_reference_easy_action_within_2_percent',
                completion_gain_mass=None, switched=0, action_counts=None, risk=None)
        a,k = np.unravel_index(np.where(safe,weights[:,None]*g,-np.inf).argmax(),g.shape)
        choice[a] = k
    use = choice>=0; pred = floor.copy()
    pred[use] = proposals[choice[use],row[use]]
    risk = policy(pred,floor,floor,target,valid,np.full(len(floor),recording),easy_cut=easy_cut,
        hard_cut=hard_cut)['recordings'][recording]
    if not risk['supported'] or risk['unknown_selected'] or risk['selected']!=int(use.sum()):
        raise ValueError('Original evaluator must accept the real, known-path witness')
    return dict(feasible=True,reason='analytic_optimum',completion_gain_mass=float(weights[use]@g[row[use],choice[use]]),
        switched=int(use.sum()),action_counts=np.bincount(choice[use],minlength=len(proposals)).tolist(),risk=risk)
