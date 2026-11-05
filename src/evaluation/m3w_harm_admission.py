"""Attribute changed fitting admissions; targets never become inference features."""
import numpy as np
from src.evaluation.m3w_decision_semantics import joint_target


def analyze(before, after, target, allowed, observed, weights):
    before, after, target = (np.asarray(x) for x in (before, after, target))
    if before.shape != target.shape[:2] or after.shape != before.shape or target.shape[-1] != 2:
        raise ValueError('Aligned proposal scores and two target channels required')
    if allowed.shape != before.shape or observed.shape != (len(before),) or weights.shape != observed.shape:
        raise ValueError('Aligned support, observation and query weights required')
    if allowed.sum(1).max(initial=0) > 1:
        raise ValueError('One frozen proposal per row required')
    if not all(np.isfinite(x).all() for x in (before, after, target, weights)) or (weights < 0).any():
        raise ValueError('Finite scores and nonnegative weights required')
    selected = allowed & observed[:, None]
    y = joint_target(target)
    positive = np.maximum(y, 0)
    negative = np.maximum(-y, 0)
    unsafe = selected & (y > 0)
    safe = selected & (y <= 0)
    accepted0, accepted1 = before <= 0, after <= 0
    q = [0., .5, .9, .95, .99, 1.]

    def quantiles(values):
        return None if not len(values) else dict(zip(map(str, q), map(float, np.quantile(values, q))))

    mass_p = float((positive * weights[:, None] * selected).sum())
    mass_n = float((negative * weights[:, None] * selected).sum())
    masks = dict(new_harmful=unsafe & ~accepted0 & accepted1,
        repaired_harmful=unsafe & accepted0 & ~accepted1,
        retained_harmful=unsafe & accepted0 & accepted1,
        new_useful=safe & ~accepted0 & accepted1,
        lost_useful=safe & accepted0 & ~accepted1)
    slices = {key: dict(count=int(mask.sum()),
        weighted_cost=float((np.abs(y)*weights[:, None]*mask).sum()),
        target_quantiles=quantiles(y[mask])) for key, mask in masks.items()}
    return dict(observed_proposals=int(selected.sum()), unsafe_proposals=int(unsafe.sum()),
        target_safe_proposals=int(safe.sum()), positive_target_mass=mass_p,
        negative_target_mass=mass_n,
        negative_to_positive_mass_ratio=mass_n/mass_p if mass_p > 0 else None,
        positive_target_quantiles=quantiles(y[unsafe]), slices=slices,
        before_false_safe=int((unsafe & accepted0).sum()), after_false_safe=int((unsafe & accepted1).sum()),
        cost_change=sum(slices[k]['weighted_cost']*sign for k, sign in
            (('new_harmful', 1), ('repaired_harmful', -1), ('new_useful', -1), ('lost_useful', 1))),
        cost_scale='sum_channels_as_parent_attribution_not_mean_channel_training_loss',
        target_safe_is_not_deployment_certificate=True)
