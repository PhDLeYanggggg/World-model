"""Label-aware mutually exclusive action oracle; never a deployable policy."""
import hashlib
import numpy as np

from src.world_model.m3w_scaled_risk_readout import policy_score
from src.evaluation.m3w_easy_constrained_policy import budget_status


def score_choices(y, envelope, choices):
    n, k, _ = y.shape
    choices = np.asarray(choices)
    if choices.shape != (n,) or not np.issubdtype(choices.dtype, np.integer) or ((choices < -1)|(choices >= k)).any():
        raise ValueError('One action or abstention per row required')
    selected = choices >= 0
    index = np.where(selected, choices, 0)
    return policy_score(y[np.arange(n), index], selected, envelope[np.arange(n), index])


def solve_action_bank(targets, envelope, allowed):
    y, env, allowed = np.asarray(targets, float), np.asarray(envelope, float), np.asarray(allowed)
    if (y.ndim != 3 or y.shape[-1] != 5 or y.shape[1] < 1 or env.shape != y.shape[:2]
            or allowed.shape != y.shape[:2] or allowed.dtype != bool):
        raise ValueError('Five aligned moments per fixed causal action and Boolean eligibility required')
    n, k, _ = y.shape
    for action in range(k):
        policy_score(y[:, action], np.zeros(n, bool), env[:, action])
        for column in (2, 3):
            np.testing.assert_array_equal(y[:, action, column], y[:, 0, column])
        np.testing.assert_array_equal(np.isfinite(y[:, action]).all(1), np.isfinite(y[:, 0]).all(1))
    valid = allowed & np.isfinite(y).all(2)
    benefit = np.where(valid, y[:, :, 0], -np.inf)
    best = benefit.argmax(1)
    choices = np.where(benefit[np.arange(n), best] > 0, best, -1)
    easy = valid & (y[:, :, 3] > 0)
    reason, anchor = None, None
    selected_easy = (choices >= 0)&(y[np.arange(n), np.where(choices >= 0, choices, 0), 3] > 0)
    if not easy.any():
        reason = 'no_real_allowed_known_easy_reference'
    elif not selected_easy.any():
        ratio = np.divide(y[:, :, 4], y[:, :, 3], out=np.full((n, k), np.inf), where=y[:, :, 3] > 0)
        safe = easy & (ratio <= .02+1e-12)
        if not safe.any():
            reason = 'no_real_easy_anchor_within_budget'
        else:
            costs = np.where(safe, y[:, :, 1], np.inf)
            row, action = np.unravel_index(costs.argmin(), costs.shape)
            choices[row] = action
            anchor = dict(row=int(row), action=int(action))
    if reason:
        choices[:] = -1
    score = score_choices(y, env, choices)
    feasible = budget_status(score)['supported']
    if feasible != (reason is None):
        raise ValueError('Oracle witness disagrees with unchanged supported budgets')
    return dict(rows=n, fixed_actions=k, real_allowed_action_occurrences=int(allowed.sum()),
        known_positive_row_count=int(np.any(valid & (y[:, :, 0] > 0), axis=1).sum()),
        budget_feasible=feasible, infeasibility_reason=reason, anchor=anchor,
        positive_feasible=feasible and score['selected_net_gain_lower_mass'] > 0,
        optimal_gain_lower_mass=score['selected_net_gain_lower_mass'] if feasible else None,
        policy=score, choice_sha256=hashlib.sha256(choices.tobytes()).hexdigest(),
        label_aware_not_deployable=True, scene_joint_conflicts_modeled=False), choices
