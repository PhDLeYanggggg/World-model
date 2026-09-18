"""Exact current-mask-input ambiguity, not a population irreducibility claim."""
import hashlib
import json
import numpy as np


def input_keys(geometry, coverage, radius, rotation, support):
    arrays=[np.asarray(x) for x in (geometry,coverage,radius,rotation,support)]
    n=len(arrays[0])
    if not n or any(len(x)!=n or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Aligned finite actual observed-input arrays required')
    keys=[]
    for i in range(n):
        h=hashlib.sha256(b'geometry_coverage_restore_mask_only_v1')
        for array in arrays:
            row=np.array(array[i],copy=True)
            if np.issubdtype(row.dtype,np.floating):
                row[row==0]=0
            h.update(json.dumps([str(row.dtype),row.shape],separators=(',',':')).encode())
            h.update(np.ascontiguousarray(row).tobytes())
        keys.append(h.hexdigest())
    return np.asarray(keys)


def collision_summary(keys,target):
    keys,target=np.asarray(keys),np.asarray(target,float)
    if keys.ndim!=1 or target.shape!=(len(keys),12,2) or not np.isfinite(target).all():
        raise ValueError('Aligned training targets required')
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    cv=np.linalg.norm(target,axis=-1).mean(1)
    zero=cv==0
    rows=[]
    covered=np.zeros(len(keys),bool)
    for group,count in enumerate(counts):
        if count<2:
            continue
        mask=inverse==group
        y=target[mask].copy();y[y==0]=0
        distinct=len(np.unique(y.reshape(count,-1),axis=0))
        nz=int(zero[mask].sum())
        conflict=distinct>1
        sufficient=bool(conflict and 2*nz>=int(count))
        if sufficient:
            covered[mask]=True
        rows.append(dict(rows=int(count),distinct_future_paths=distinct,zero_paths=nz,
            conflicting_labels=conflict,zero_is_empirical_ade_optimum=sufficient,
            baseline_cost_sum=float(cv[mask].sum())))
    return dict(training_rows=len(keys),unique_observed_inputs=len(unique),duplicate_groups=len(rows),
        rows_in_duplicate_groups=int(counts[counts>1].sum()),
        conflicting_groups=sum(r['conflicting_labels'] for r in rows),
        rows_in_conflicting_groups=sum(r['rows'] for r in rows if r['conflicting_labels']),
        zero_optimal_conflicting_groups=sum(r['zero_is_empirical_ade_optimum'] for r in rows),
        rows_in_zero_optimal_conflicting_groups=int(covered.sum()),
        baseline_cost_fraction_in_zero_optimal_conflicts=float(cv[covered].sum()/cv.sum()) if cv.sum()>0 else None,
        largest_duplicate_group=int(counts.max()) if len(counts) else 0,
        duplicate_group_details=rows)
