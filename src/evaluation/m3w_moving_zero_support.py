"""Causal-neighborhood diagnostics, never a fitted or deployed risk threshold."""
import hashlib
import numpy as np


def fixed_controls(ids, recordings, frames, tracks, count):
    ids=np.asarray(ids)
    if ids.ndim != 1 or count < 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique one-dimensional eligible ids and positive count required')
    keys=[hashlib.sha256(f'{recordings[i]}|{int(frames[i])}|{tracks[i]}'.encode()).hexdigest() for i in ids]
    return ids[np.argsort(np.asarray(keys),kind='stable')[:count]]


def neighbors(source, query, source_ids, *, k=512, block=4096):
    """Direct float64 distances avoid subtracting nearly equal squared norms."""
    x,q,ids=np.asarray(source),np.asarray(query),np.asarray(source_ids)
    if (x.ndim != 2 or q.shape != (x.shape[1],) or ids.shape != (len(x),)
            or not len(x) or len(np.unique(ids)) != len(ids) or k < 1 or block < 1
            or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError('Finite aligned source-only features and query required')
    d=np.empty(len(x),np.float64); exact=np.zeros(len(x),bool)
    for start in range(0,len(x),block):
        z=x[start:start+block].astype(np.float64)-q.astype(np.float64)
        d[start:start+block]=np.einsum('ij,ij->i',z,z)/x.shape[1]
        exact[start:start+block]=np.all(z==0,axis=1)
    if not np.isfinite(d).all(): raise ValueError('Distance overflow')
    order=np.lexsort((ids,d))
    return dict(squared_distance=d,order=order,nearest=order[:k],exact=exact)


def describe_neighbors(result, event, benefit, harm, tracks, *, ks=(32,128,512)):
    d,order=result['squared_distance'],result['order']
    event,b,h,t=map(np.asarray,(event,benefit,harm,tracks))
    if (event.dtype!=bool or any(v.shape!=d.shape for v in (event,b,h,t))
            or not np.isfinite(b).all() or not np.isfinite(h).all() or (b<0).any() or (h<0).any()):
        raise ValueError('Only aligned supported source-label summaries allowed')
    positive=np.flatnonzero(event[order]); rank=int(positive[0]+1) if len(positive) else None
    exact=result['exact']; subsets={}
    for k in ks:
        ii=order[:k]; nt=len(np.unique(t[ii]))
        subsets[str(k)]=dict(rows=len(ii),tracks=nt,zero_rows=int(event[ii].sum()),
            beneficial_rows=int((b[ii]>h[ii]).sum()),harmful_rows=int((h[ii]>b[ii]).sum()),
            zero_tracks=len(np.unique(t[ii][event[ii]])),mean_relative_gain=float((b[ii]-h[ii]).mean()))
    return dict(nearest_distance=float(np.sqrt(d[order[0]])),nearest_zero_rank=rank,
        nearest_zero_rank_fraction=rank/len(d) if rank is not None else None,
        nearest_zero_distance=float(np.sqrt(d[order[rank-1]])) if rank is not None else None,
        exact_rows=int(exact.sum()),exact_zero_rows=int((exact&event).sum()),
        exact_nonzero_rows=int((exact&~event).sum()),neighborhoods=subsets)
