"""Versioned raw reader accepting the two observed official export schemas."""
from collections import Counter

import numpy as np
import pandas as pd

from src.evaluation.m3w_european_squares_intake import COLUMNS, DTYPE


def read_raw_csv(stream, chunksize=250000):
    pieces, names = [], Counter()
    previous_frame = -1
    schema = None
    for chunk in pd.read_csv(stream, chunksize=chunksize):
        columns = tuple(chunk.columns)
        if columns not in (COLUMNS, tuple(c for c in COLUMNS if c != 'class_name')):
            raise ValueError('Unexpected raw tracker schema')
        if schema is not None and columns != schema:
            raise ValueError('Raw schema changes within recording')
        schema = columns
        numeric = chunk.drop(columns='class_name', errors='ignore').to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError('Nonfinite raw field')
        result = np.empty(len(chunk), dtype=DTYPE)
        for target, source in (('agent', 'tracker_id'), ('frame', 'frame_index'), ('class_id', 'class_id')):
            values = chunk[source].to_numpy(dtype=np.float64)
            if np.any(values < 0) or np.any(values > 2**53-1) or np.any(values != np.floor(values)):
                raise ValueError('Invalid raw integer')
            result[target] = values.astype(np.int64)
        for name in ('x_min', 'y_min', 'x_max', 'y_max', 'confidence'):
            result[name] = chunk[name].to_numpy(dtype=np.float64)
        if (np.any(result['x_max'] < result['x_min']) or np.any(result['y_max'] < result['y_min'])
                or np.any(result['confidence'] < 0) or np.any(result['confidence'] > 1)):
            raise ValueError('Invalid raw box or confidence')
        if len(result) and (result['frame'][0] < previous_frame or np.any(np.diff(result['frame']) < 0)):
            raise ValueError('Raw file not in frame order')
        if len(result):
            previous_frame = int(result['frame'][-1])
        if 'class_name' in chunk:
            if chunk['class_name'].isna().any():
                raise ValueError('Null supplied class name')
            names.update(map(str, chunk['class_name']))
        else:
            # Preserve absence rather than inferring a human-verified person label.
            names['not_provided'] += len(chunk)
        pieces.append(result)
    if not pieces or not sum(map(len, pieces)):
        raise ValueError('Empty raw recording')
    rows = np.concatenate(pieces)
    rows.sort(order=['agent', 'frame'])
    if np.any((rows['agent'][1:] == rows['agent'][:-1]) & (rows['frame'][1:] == rows['frame'][:-1])):
        raise ValueError('Duplicate scoped agent/frame; recording quarantined, not silently deduplicated')
    return rows, dict(names)
