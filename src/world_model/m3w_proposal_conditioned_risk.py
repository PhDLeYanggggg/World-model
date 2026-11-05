"""Proposal-conditioned cost supervision with an unchanged original risk veto."""
import numpy as np

from src.world_model import m3w_amplitude_cost as amplitude
from src.world_model import m3w_moving_deferral as router
from src.world_model.m3w_source_separated_critics import matched_self

base = router.base


def condition(data, logits):
    c = data['causal']
    if set(c) != base.CAUSAL_KEYS:
        raise ValueError('Only the declared causal schema may define training proposals')
    proposal, _ = router.choose(logits, c['allowed'])
    mask = np.zeros_like(c['allowed'])
    rows = np.flatnonzero(proposal >= 0)
    mask[rows, proposal[rows]] = True
    # Keep every row and query, including router fallback and unknown labels.
    return dict(data, causal=dict(c, allowed=mask)), proposal


def assert_excluded(state, causal, source_names):
    if (set(state['fit_ids']) & set(causal['ids'])
            or set(state['preprocess']['source_names']) & set(source_names)
            or set(state['source_sites']) & set(source_names)):
        raise ValueError('Proposer and risk fitting/validation sources must be disjoint')


def fit(data, **kwargs):
    return amplitude.fit(data, arm='amplitude_bank', **kwargs)


def predict(state, causal):
    return amplitude.predict(state, causal)


def causal_choices(logits, all_action_costs, conditioned_costs, causal):
    proposal, utility = router.choose(logits, causal['allowed'])
    result = dict(physical_floor=np.full(len(proposal), -1, np.int64), public_router=proposal)
    rows = np.flatnonzero(proposal >= 0)
    for name, prediction in (('all_action', all_action_costs), ('conditioned', conditioned_costs)):
        p = np.asarray(prediction)
        if p.shape != (len(proposal), 7, 5) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Seven finite nonnegative five-moment predictions required')
        cost = p[rows, proposal[rows]]
        keep = ((cost[:, 2] > 0) & (cost[:, 3] > 0)
            & (cost[:, 1] <= .02*cost[:, 2]) & (cost[:, 4] <= .02*cost[:, 3]))
        veto = np.full(len(proposal), -1, np.int64)
        veto[rows[keep]] = proposal[rows[keep]]
        result[name+'_veto'] = veto
        result[name+'_matched'] = matched_self(proposal, utility, veto, causal['groups'], causal['ids'])
    return result
