"""Replay immutable older choices without pretending to re-infer their costs."""
import hashlib
import json

import numpy as np

POLICIES=('independent','frame_uniform','joint_matched','joint_free')


def binding(value):
    x=np.ascontiguousarray(value)
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()


def verified_choices(path,expected_sha256,causal,packet,seconds):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('Old solver ledger hash changed')
    with np.load(path,allow_pickle=False)as cache:
        meta=json.loads(str(cache['metadata_json']))
        choices={k:cache[k].copy()for k in POLICIES}
    expected={k:binding(v)for k,v in zip(('allowed','recordings','frames','ids'),
        (causal['allowed'],packet['recordings'],packet['frames'],packet['ids']))}
    if (any(meta['bindings'].get(k)!=v for k,v in expected.items())
            or meta['bindings'].get('seconds')!=seconds or meta['bindings'].get('version')!=1
            or meta['completed_queries']!=len(causal['groups'])):
        raise ValueError('Old choices must match the complete current causal input roster and budget')
    n=len(packet['ids']);allowed=causal['allowed']
    for choice in choices.values():
        if choice.shape!=(n,)or choice.dtype.kind not in 'iu' or np.any((choice< -1)|(choice>=7)):
            raise ValueError('Aligned integer action choices required')
        at=np.flatnonzero(choice>=0)
        if not allowed[at,choice[at]].all():raise ValueError('Old action no longer legal')
    for at in causal['groups']:
        if (choices['independent'][at]>=0).sum()!=(choices['joint_matched'][at]>=0).sum():
            raise ValueError('Same-count control changed')
    return choices,dict(result_source='cached_verified_choices_fresh_scalar_replay_required',
        sha256=expected_sha256,current_input_bindings_verified=True,
        old_cost_prediction_freshly_reinferred=False,old_predicted_feasibility_freshly_rechecked=False,
        bitwise_MILP_reoptimization_claimed=False,original_prediction_binding=meta['bindings']['prediction'])
