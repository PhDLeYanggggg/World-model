"""Aggregate input clipping diagnostics; labels never define causal regions."""
import numpy as np


def profile(x, mean, std, clip, regions, observed, moments, *, chunk=1024):
    x, mean, std = np.asarray(x), np.asarray(mean), np.asarray(std)
    observed, moments = np.asarray(observed), np.asarray(moments)
    n, d = x.shape if x.ndim == 2 else (0, 0)
    if (not n or not d or mean.shape != (d,) or std.shape != (d,)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()
            or not np.isfinite(clip) or clip <= 0 or chunk <= 0
            or observed.shape != (n,) or observed.dtype != bool
            or moments.shape != (n, 5) or not np.isfinite(moments).all()
            or (moments < 0).any() or not regions):
        raise ValueError('Aligned finite inputs, original normalization and separate diagnostic labels required')
    regions = {k:np.asarray(v) for k,v in regions.items()}
    if any(v.shape != (n,) or v.dtype != bool for v in regions.values()):
        raise ValueError('Each fixed causal region must be an aligned Boolean mask')
    bins = ((0, 0), (1, 4), (5, 16), (17, d))
    outputs = {name:dict(rows=int(mask.sum()), clipped_feature_counts=np.zeros(d, np.int64),
        maximum_absolute_standardized_value=0., bins=[dict(minimum=a, maximum=b, rows=0,
        known_rows=0, known_moments=np.zeros(5), all_bound_moments=np.zeros(5)) for a,b in bins])
        for name,mask in regions.items()}
    for start in range(0,n,chunk):
        stop = min(n,start+chunk)
        z = (x[start:stop].astype(float)-mean)/std
        clipped = np.abs(z) > clip
        counts = clipped.sum(1)
        for name,mask in regions.items():
            take = mask[start:stop]
            out = outputs[name]
            if not take.any():
                continue
            out['clipped_feature_counts'] += clipped[take].sum(0)
            out['maximum_absolute_standardized_value'] = max(
                out['maximum_absolute_standardized_value'],float(np.abs(z[take]).max()))
            for cell,(lo,hi) in zip(out['bins'],bins):
                at = take & (counts >= lo) & (counts <= hi)
                known = at & observed[start:stop]
                cell['rows'] += int(at.sum())
                cell['known_rows'] += int(known.sum())
                cell['known_moments'] += moments[start:stop][known].sum(0)
                cell['all_bound_moments'] += moments[start:stop][at].sum(0)
    for out in outputs.values():
        if sum(cell['rows'] for cell in out['bins']) != out['rows']:
            raise ValueError('Clipping bins must preserve every causal row')
        out['clipped_feature_counts'] = out['clipped_feature_counts'].tolist()
        for cell in out['bins']:
            for key in ('known_moments','all_bound_moments'):
                cell[key] = cell[key].tolist()
    return outputs
