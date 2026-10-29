"""Condition only the positive-support dual; preserve the original feasible set."""
from contextlib import contextmanager
import torch
from src.world_model import m3w_multiaction_record_policy as base


def support_scaled_pass(original, model, data, dual, temperature, chunk_size, *, backward=False):
    unit = data['support_min']
    if not torch.isfinite(unit).all() or (unit <= 0).any():
        raise ValueError('Positive genuine fitting support requirement needed')
    factor = data['risk_scales'][:, 1]/unit
    if not torch.isfinite(factor).all():
        raise ValueError('Finite training-only support unit ratio required')
    effective = torch.cat((dual[:, :2], (dual[:, 2]*factor)[:, None]), 1)
    result = original(model, data, effective, temperature, chunk_size, backward=backward)
    # Original exact costs, actions and all/easy constraints are unchanged.
    result['excess'] = torch.cat((result['excess'][:, :2],
        ((unit-result['support'])/unit)[:, None]), 1).float()
    return result


@contextmanager
def installed(enabled=True):
    original = base.full_pass
    if enabled:
        def conditioned(*args, **kwargs):
            return support_scaled_pass(original, *args, **kwargs)
        base.full_pass = conditioned
    try:
        yield
    finally:
        base.full_pass = original
