"""Role-safe donor policies and label-blind allocation at the frozen budget."""

import numpy as np

from src.evaluation import m3w_count_motion_capacity as capacity
from src.world_model.m3w_recording_count import streaming_count


def fit_roster(sites, omitted):
    sites, omitted = tuple(sites), tuple(omitted)
    if (len(set(sites)) != len(sites) or len(set(omitted)) != len(omitted)
            or not set(omitted) <= set(sites)):
        raise ValueError('Invalid exclusion roster')
    result = sorted(set(sites)-set(omitted))
    if not result:
        raise ValueError('No remaining training sites')
    return result


def verify_role(identity, sites, omitted):
    if (identity.get('omitted') != sorted(omitted)
            or identity.get('fit') != fit_roster(sites, omitted)):
        raise ValueError('Producer exposed an excluded location')


def score_anchor(logits, legal, movement, recordings, starts):
    """Match the evaluation score allocator without accepting outcome arrays."""
    if len(set(recordings)) != 1:
        raise ValueError('Allocate one recording at a time')
    online = streaming_count(logits, legal, recordings, starts, .2)
    ix = np.arange(len(online))
    cap = float(movement[ix, online].sum())
    count, status = capacity.allocate(legal, movement, int(online.sum()), cap, score=logits)
    if count is not None:
        if (count.shape != online.shape or not np.issubdtype(count.dtype, np.integer)
                or (count < 0).any() or (count > 2).any() or count.sum() != online.sum()
                or not legal[ix, count].all() or not capacity.respects_tokens(count, legal[:, 1])
                or not capacity.within_cap(movement[ix, count].sum(), cap)
                or logits[ix, count].sum(dtype=float)+1e-8 < logits[ix, online].sum(dtype=float)):
            raise ValueError('Score allocation changed the budget or lost to its feasible anchor')
    return count, cap, status


def total_fields(rows, fields):
    """Retain zero and signed values, unlike addition of Counters."""
    return {field: sum(row.get(field, 0) for row in rows) for field in fields}
