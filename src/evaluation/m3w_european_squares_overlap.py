"""Detection-geometry fingerprints for shifted, re-ID'd partial clip screening."""
import hashlib

import numpy as np

BLOCK_DTYPE=np.dtype([('frame','<i8'),('exact','V32'),('quantized','V32'),
                      ('exact_valid','?'),('quantized_valid','?')])


def frame_blocks(rows, length=8):
    if length<2:
        raise ValueError('At least two frames required')
    fields=('class_id','x_min','y_min','x_max','y_max')
    order=np.lexsort(tuple(rows[k] for k in reversed(('frame',*fields))))
    ordered=rows[order]
    frames,starts,sizes=np.unique(ordered['frame'],return_index=True,return_counts=True)
    signatures={mode:np.empty(len(frames),dtype='V32') for mode in ('exact','quantized')}
    geometry=np.column_stack([ordered[k] for k in fields]).astype('<f8')
    for i,(start,size) in enumerate(zip(starts,sizes)):
        value=geometry[start:start+size]
        exact=value.tobytes()
        quantized_values=np.rint(value).astype('<i8')
        quantized_order=np.lexsort(tuple(quantized_values[:,k] for k in range(len(fields)-1,-1,-1)))
        quantized=quantized_values[quantized_order].tobytes()
        signatures['exact'][i]=hashlib.sha256(exact).digest()
        signatures['quantized'][i]=hashlib.sha256(quantized).digest()
    stats=dict(rows=len(rows),nonempty_frames=len(frames),consecutive_blocks=0,
               exact_dynamic_blocks=0,quantized_dynamic_blocks=0,
               length=length,geometry_uses_tracker_ids=False)
    if len(frames)<length:
        return np.empty(0,dtype=BLOCK_DTYPE),stats
    grid=np.lib.stride_tricks.sliding_window_view(frames,length)
    valid=np.flatnonzero((np.diff(grid,axis=1)==1).all(axis=1))
    out=np.zeros(len(valid),dtype=BLOCK_DTYPE)
    out['frame']=frames[valid]
    stats['consecutive_blocks']=len(out)
    for mode,signature in signatures.items():
        windows=np.lib.stride_tricks.sliding_window_view(signature,length)
        dynamic=np.any(windows[valid]!=windows[valid,:1],axis=1)
        out[mode+'_valid']=dynamic
        for j in np.flatnonzero(dynamic):
            out[mode][j]=hashlib.sha256(windows[valid[j]].tobytes()).digest()
        stats[mode+'_dynamic_blocks']=int(dynamic.sum())
    return out,stats


def duplicate_groups(records, mode):
    if mode not in ('exact','quantized'):
        raise ValueError('Unknown signature mode')
    dtype=np.dtype([('signature','V32'),('record','<i4'),('frame','<i8')])
    parts=[]
    names=[name for name,_ in records]
    if len(set(names))!=len(names):
        raise ValueError('Repeated source recording')
    for i,(_,blocks) in enumerate(records):
        selected=blocks[blocks[mode+'_valid']]
        part=np.empty(len(selected),dtype=dtype)
        part['signature']=selected[mode]
        part['frame']=selected['frame']
        part['record']=i
        parts.append(part)
    if not parts:
        return []
    joined=np.concatenate(parts)
    if not len(joined):
        return []
    joined.sort(order=['signature','record','frame'])
    signature=joined['signature']
    starts=np.r_[0,np.flatnonzero(signature[1:]!=signature[:-1])+1]
    ends=np.r_[starts[1:],len(joined)]
    output=[]
    for start,end in zip(starts[ends-starts>1],ends[ends-starts>1]):
        group=joined[start:end]
        if len(np.unique(group['record']))<2:
            continue
        # Repeated instances within a recording remain visible, but cannot on
        # their own become cross-recording evidence.
        pairs=sorted(set((int(r['record']),int(r['frame'])) for r in group))
        output.append([dict(recording=names[i],start_frame=frame) for i,frame in pairs])
    return output
