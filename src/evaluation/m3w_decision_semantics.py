"""Separate proposal-support, channel-sign and joint-decision diagnostics."""
import numpy as np


def proposal_mask(proposal, allowed):
    proposal, allowed = np.asarray(proposal), np.asarray(allowed)
    if (allowed.dtype != bool or allowed.ndim != 2 or proposal.shape != (len(allowed),)
            or not np.issubdtype(proposal.dtype, np.integer)
            or (proposal < -1).any() or (proposal >= allowed.shape[1]).any()):
        raise ValueError('Aligned legal causal proposals required')
    rows = np.flatnonzero(proposal >= 0)
    if not allowed[rows, proposal[rows]].all():
        raise ValueError('Router proposal must be legal')
    mask = np.zeros_like(allowed)
    mask[rows, proposal[rows]] = True
    return mask


def joint_target(channel_target):
    """Supervision only: a positive channel makes the conjunction unsafe."""
    y = np.asarray(channel_target)
    if y.shape[-1] != 2 or not np.isfinite(y).all():
        raise ValueError('Two finite normalized channel targets required')
    positive = np.maximum(y, 0).sum(-1)
    negative = np.maximum(-y, 0).sum(-1)
    return np.where(positive > 0, positive, -negative)


def diagnose(risk, decision, target, allowed, observed, weights):
    risk, decision, target = map(np.asarray, (risk, decision, target))
    if (risk.shape != target.shape or decision.shape != target.shape or target.shape[-1] != 2
            or allowed.shape != target.shape[:2] or observed.shape != (len(target),)
            or weights.shape != (len(target),) or (allowed.sum(1) > 1).any()
            or not all(np.isfinite(v).all() for v in (risk, decision, target, weights))):
        raise ValueError('Aligned proposal-conditioned supervision and predictions required')
    mask = allowed & observed[:, None]
    mixed = (target.min(-1) < 0) & (target.max(-1) > 0)
    safe = (target <= 0).all(-1)
    zero = (target == 0).all(-1)
    labels = joint_target(target)
    scores = np.maximum(risk, decision).max(-1)
    accept = scores <= 0
    w = weights[:, None] * mask
    false_safe, false_reject = accept & ~safe, ~accept & safe
    cost = w * np.abs(labels)
    return dict(known_proposals=int(mask.sum()), unknown_proposals=int((allowed & ~observed[:, None]).sum()),
        mixed_sign_proposals=int((mask & mixed).sum()),
        true_joint_safe_proposals=int((mask & safe).sum()), zero_target_proposals=int((mask & zero).sum()),
        accepted_known_proposals=int((mask & accept).sum()),
        false_safe_count=int((mask & false_safe).sum()), false_reject_count=int((mask & false_reject).sum()),
        false_safe_cost=float((cost*false_safe).sum()), false_reject_cost=float((cost*false_reject).sum()),
        total_joint_cost=float((cost*(false_safe | false_reject)).sum()),
        mixed_sign_weighted_fraction=float((w*mixed).sum()/w.sum()) if w.sum() else None,
        nonjoint_channel_reward_mass_on_joint_unsafe=float((w*np.maximum(-target, 0).sum(-1)*~safe).sum()),
        joint_unsafe_penalty_mass=float((w*np.maximum(target, 0).sum(-1)).sum()),
        descriptive_fitting_only=True, population_risk_certificate=False)
