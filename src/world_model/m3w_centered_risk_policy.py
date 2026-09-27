"""Fixed signed-score offsets with an equal-count, uncentered-risk control."""
import numpy as np
from src.world_model.m3w_query_utility import grouped

POLICIES = ('centered_independent', 'centered_joint', 'matched_raw_joint',
            'centered_utility_topk', 'centered_query_uniform')


def decisions(utility, signed_risk, offset, eligible, recordings, frames, ids, *, node_limit=256):
    risk, offset = np.asarray(signed_risk, float), np.asarray(offset, float)
    eligible = np.asarray(eligible)
    if (risk.shape != (len(eligible), 2) or offset.shape != (2,)
            or not np.isfinite(risk).all() or not np.isfinite(offset).all()
            or (offset < 0).any() or eligible.dtype != bool):
        raise ValueError('Two finite nonnegative fitting-only offsets and causal signed scores required')
    centered = risk+offset
    anchor = eligible & (centered <= 0).all(1)
    original = eligible & (risk <= 0).all(1)
    assert not (anchor & ~original).any()
    fresh, stats = grouped(utility, centered, eligible, anchor, recordings, frames, ids, node_limit=node_limit)
    control, control_stats = grouped(utility, risk, eligible, anchor, recordings, frames, ids, node_limit=node_limit)
    return dict(centered_independent=anchor, centered_joint=fresh['joint_utility'],
        matched_raw_joint=control['joint_utility'], centered_utility_topk=fresh['utility_topk'],
        centered_query_uniform=fresh['query_uniform']), dict(centered=stats, matched_raw=control_stats)


def check_queries(actions, raw, offset, eligible, recordings, frames, ids):
    raw = np.asarray(raw, float); centered = raw+np.asarray(offset)
    groups = {}
    for i, k in enumerate(zip(recordings, frames)):
        groups.setdefault((str(k[0]), int(k[1])), []).append(i)
    counts = dict(queries=0, zero_action_queries=0, changed_equal_count_queries=0)
    for positions in groups.values():
        ix = np.asarray(positions); a=actions['centered_independent'][ix]; k=int(a.sum())
        np.testing.assert_array_equal(a, eligible[ix] & (centered[ix] <= 0).all(1))
        for key in POLICIES:
            take=actions[key][ix]
            assert take.dtype==bool and not (take & ~eligible[ix]).any()
            if key!='centered_query_uniform':assert take.sum()==k
            if key not in ('centered_utility_topk',):
                q=raw[ix] if key=='matched_raw_joint' else centered[ix]
                scale=np.maximum(np.abs(q).max(0),1e-12)
                assert np.all((q/scale)[take].sum(0)<=1e-10)
        counts['queries']+=1; counts['zero_action_queries']+=k==0
        counts['changed_equal_count_queries']+=bool(np.any(actions['centered_joint'][ix]!=actions['matched_raw_joint'][ix]))
    return counts
