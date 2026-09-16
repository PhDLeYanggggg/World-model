"""Order-independent identity of OOF queries, causal features and producers."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def oof_feature_identity(groups):
    if not groups:
        raise ValueError('Nonempty OOF groups required')
    records, seen, dimensions = [], set(), set()
    for group in groups:
        x = np.asarray(group['features'])
        if x.ndim != 2 or x.shape[1] < 1 or len(x) != len(group['identities']) or not np.isfinite(x).all():
            raise ValueError('Aligned finite OOF features and query identities required')
        dimensions.add(x.shape[1])
        producer = {k: group[k] for k in ('predictor_id', 'predictor_sha256', 'protocol_sha256', 'baseline_name', 'metric')}
        for row, features in zip(group['identities'], x):
            if (not isinstance(row['recording_id'], str) or not row['recording_id']
                    or any(isinstance(row[k], (bool, np.bool_)) or not np.isfinite(row[k]) or int(row[k]) != row[k]
                           for k in ('agent_id', 'frame_id', 'horizon_raw'))):
                raise ValueError('Exact integer query identity required')
            key = (row['recording_id'], int(row['agent_id']), int(row['frame_id']), int(row['horizon_raw']))
            if key in seen:
                raise ValueError('Duplicated OOF query in input identity')
            seen.add(key)
            # Compare numerical features at float64 precision, independently of row ordering/dtype.
            digest = hashlib.sha256(json.dumps({'query': key, **producer}, sort_keys=True,
                                               separators=(',', ':'), allow_nan=False).encode())
            digest.update(np.asarray(features, dtype='<f8').tobytes())
            records.append((key, digest.digest()))
    if not records or len(dimensions) != 1:
        raise ValueError('Nonempty common OOF feature schema required')
    digest = hashlib.sha256(b'm3w_oof_inputs_v1')
    for _, row_digest in sorted(records):
        digest.update(row_digest)
    return {'schema': 'm3w_oof_inputs_v1', 'sha256': digest.hexdigest(),
            'rows': len(records), 'feature_dimension': dimensions.pop(),
            'includes_targets': False, 'includes_predictor_identity': True}
