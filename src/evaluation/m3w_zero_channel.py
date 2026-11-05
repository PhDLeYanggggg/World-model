"""Fitting-only attribution of conjunction vetoes, never an input feature."""
import numpy as np
from src.evaluation.m3w_decision_semantics import diagnose, joint_target


def attribute(risk, decision, target, allowed, observed, weights):
    result = diagnose(risk, decision, target, allowed, observed, weights)
    selected = allowed & observed[:, None]
    safe = (target <= 0).all(-1)
    blocks = np.stack((risk, decision), -2) > 0
    rejected = selected & safe & blocks.any((-2, -1))
    zero = target == 0
    zero_block = (blocks & zero[..., None, :]).any((-2, -1))
    nonzero_block = (blocks & ~zero[..., None, :]).any((-2, -1))
    cost = weights[:, None]*np.abs(joint_target(target))
    slices = dict(zero_only=rejected & zero_block & ~nonzero_block,
        nonzero_only=rejected & ~zero_block & nonzero_block,
        both=rejected & zero_block & nonzero_block)
    details = {}
    for h, head in enumerate(('risk', 'decision')):
        for k, channel in enumerate(('all', 'easy')):
            block = rejected & blocks[..., h, k]
            details[head+'_'+channel] = dict(rejected_rows=int(block.sum()),
                zero_target_rows=int((block & zero[..., k]).sum()),
                weighted_cost=float((cost*block).sum()),
                zero_target_cost=float((cost*block*zero[..., k]).sum()))
    return dict(**result,
        zero_channel_observed_proposals=int((selected & zero.any(-1)).sum()),
        false_reject_partition={key: dict(rows=int(mask.sum()), cost=float((cost*mask).sum()))
            for key, mask in slices.items()}, blocking_outputs=details,
        output_counts_overlap=True, zero_target_is_not_deployment_certificate=True)
