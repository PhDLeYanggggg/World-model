"""Label-aware TRAIN feasibility witnesses, never inference policies."""
import hashlib

import numpy as np

from src.world_model import m3w_scaled_risk_readout as metrics
from src.evaluation.m3w_easy_constrained_policy import budget_status


def solve(y, envelope, allowed):
    """Maximize completion-lower utility under the existing supported budgets.

    The five-moment contract has mutually exclusive benefit/harm. Every positive
    benefit is harmless; unknown and harmful actions cannot increase utility.
    If no beneficial easy reference exists, a single minimum-harm easy anchor
    with H/R <= budget is optimal. No mixture of individually excessive easy
    ratios can satisfy their aggregate ratio. This is an offline oracle only.
    """
    y, envelope, allowed = np.asarray(y, float), np.asarray(envelope, float), np.asarray(allowed)
    if allowed.shape != (len(y),) or allowed.dtype != bool:
        raise ValueError('Aligned Boolean allowed actions required')
    metrics.policy_score(y, np.zeros(len(y), bool), envelope)
    known = np.isfinite(y).all(1)
    valid = allowed & known
    benefit = valid & (y[:, 0] > 0)
    take = benefit.copy()
    easy = valid & (y[:, 3] > 0)
    anchor = None
    reason = None
    if not easy.any():
        reason = 'no_allowed_known_easy_reference'
    elif not (take & easy).any():
        candidates = np.flatnonzero(easy & (y[:, 4]/np.where(y[:, 3] > 0, y[:, 3], 1.) <= .02+1e-12))
        if len(candidates):
            anchor = int(candidates[np.argmin(y[candidates, 1])])
            take[anchor] = True
        else:
            reason = 'no_easy_anchor_within_budget'
    if reason:
        take[:] = False
    score = metrics.policy_score(y, take, envelope)
    feasible = budget_status(score)['supported']
    if feasible != (reason is None):
        raise AssertionError('Oracle witness and unchanged budgets disagree')
    result = dict(allowed_rows=int(allowed.sum()), known_allowed_rows=int(valid.sum()),
        unknown_allowed_rows=int((allowed & ~known).sum()), known_positive_rows=int(benefit.sum()),
        positive_benefit_upper_mass=float(y[benefit, 0].sum()),
        allowed_known_easy_rows=int(easy.sum()), anchor_count=int(anchor is not None),
        anchor_harm_mass=float(y[anchor, 1]) if anchor is not None else 0.,
        budget_feasible=feasible, infeasibility_reason=reason,
        optimal_gain_lower_mass=score['selected_net_gain_lower_mass'] if feasible else None,
        safe_positive_selection_exists=feasible and score['selected_net_gain_lower_mass'] > 0,
        policy=score, action_sha256=hashlib.sha256(take.tobytes()).hexdigest(),
        label_aware_diagnostic_only=True, deployable=False)
    return result, take


def evaluate(args, eligible, policies):
    _, env, y, _, _, recordings, _, _ = args
    if not policies or np.asarray(recordings).shape != (len(y),):
        raise ValueError('Frozen policies and aligned recordings required')
    recordings = np.asarray(recordings).astype(str)
    oracle, _ = solve(y, env, eligible)
    details = {}
    for recording in np.unique(recordings):
        mask = recordings == recording
        details[str(recording)] = dict(oracle=solve(y[mask], env[mask], eligible[mask])[0], policies={})
    out = {}
    for name, action in policies.items():
        if action.dtype != bool or action.shape != eligible.shape or (action & ~eligible).any():
            raise ValueError('Frozen actions cannot bypass causal eligibility')
        score = metrics.policy_score(y, action, env)
        repair, _ = solve(y, env, action)
        for recording, item in details.items():
            mask = recordings == recording
            item['policies'][name] = dict(observed=metrics.policy_score(y[mask], action[mask], env[mask]),
                deletion_oracle=solve(y[mask], env[mask], action[mask])[0])
        selected_positive = float(y[action & np.isfinite(y).all(1), 0].sum())
        for key in ('known_rows', 'unknown_rows', 'selected_count', 'selected_unknown'):
            if sum(d['policies'][name]['observed'][key] for d in details.values()) != score[key]:
                raise AssertionError('Recording accounting must reproduce the complete view')
        out[name] = dict(observed=score, deletion_oracle=repair,
            retained_positive_benefit_mass=selected_positive,
            missed_positive_benefit_mass=oracle['positive_benefit_upper_mass']-selected_positive,
            all_record_deletion_feasible=all(d['policies'][name]['deletion_oracle']['budget_feasible']
                                            for d in details.values()))
    known = np.isfinite(y).all(1)
    return dict(rows=len(y), eligible_rows=int(eligible.sum()), known_rows=int(known.sum()),
        full_known_reference_mass=float(y[known, 2].sum()),
        oracle=oracle, recordings=details, policies=out,
        all_recordings_feasible=all(r['oracle']['budget_feasible'] for r in details.values()),
        all_recordings_positive=all(r['oracle']['safe_positive_selection_exists'] for r in details.values()),
        label_aware_diagnostic_only=True, deployment_changed=False)
