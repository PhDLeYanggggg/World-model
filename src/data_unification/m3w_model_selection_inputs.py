"""Input-only adapter for frozen European model-selection recordings."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.data_unification import m3w_european_squares_source as source

INPUT_FIELDS = source.INPUT_FIELDS


def require_input_role(manifest, member):
    if manifest.get('status') != 'restricted_source_training_admitted_reserved_roles_closed':
        raise ValueError('Frozen source role admission required')
    matches = [r for r in manifest['recordings'] if r['source_member'] == member]
    if len(matches) != 1:
        raise ValueError('Unique recording identity required')
    row = matches[0]
    if (row['role'] != 'model_selection_reserved' or row['training_access']
            or row['predictive_readout_access']):
        raise PermissionError('Only closed model-selection inputs are admitted here')
    return row


def build_inputs(rows, query_frames):
    query_frames = np.asarray(query_frames)
    if (query_frames.ndim != 1 or not np.issubdtype(query_frames.dtype, np.integer)
            or (query_frames < 0).any() or (np.diff(query_frames) <= 0).any()):
        raise ValueError('Increasing nonnegative integer query frames required')
    agents, starts, sizes = np.unique(rows['agent'], return_index=True, return_counts=True)
    if not np.array_equal(rows['agent'], np.repeat(agents, sizes)):
        raise ValueError('Agent/frame-sorted raw rows required')
    tracks = {int(a): rows[s:s+n] for a,s,n in zip(agents, starts, sizes)}
    if any((np.diff(track['frame']) <= 0).any() for track in tracks.values()):
        raise ValueError('Repeated or unordered track frames')
    order = np.argsort(rows['frame'], kind='stable')
    frames = rows['frame'][order]
    lo = np.searchsorted(frames, query_frames)
    hi = np.searchsorted(frames, query_frames, side='right')
    offsets = np.r_[0, np.cumsum(hi-lo)].astype('<i8')
    n = int(offsets[-1]); k = source.HISTORY
    values = dict(query_frame=np.empty(n,dtype='<i8'), agent_id=np.empty(n,dtype='<i8'),
        class_id=np.empty(n,dtype='<i8'), history_xy=np.zeros((n,k,2)),
        history_boxes=np.zeros((n,k,4)), history_valid=np.zeros((n,k),bool),
        velocity_causal_fd=np.zeros((n,k,2)), velocity_valid=np.zeros((n,k),bool),
        target_eligible=np.zeros(n,bool), baseline_cv=np.zeros((n,source.FUTURE,2)),
        query_offsets=offsets)
    for qi,query in enumerate(query_frames):
        current = np.sort(rows[order[lo[qi]:hi[qi]]], order='agent')
        grid = query-np.arange(k-1,-1,-1)*source.STRIDE
        for j,row in enumerate(current):
            dest = int(offsets[qi])+j
            boxes,valid = source.lookup(tracks[int(row['agent'])],grid)
            xy = (boxes[:,:2]+boxes[:,2:])/2
            vv = np.r_[False,valid[1:] & valid[:-1]]
            velocity = np.zeros_like(xy)
            velocity[1:] = np.where(vv[1:,None],np.diff(xy,axis=0)/source.STRIDE,0.)
            values['query_frame'][dest] = query
            values['agent_id'][dest] = row['agent']
            values['class_id'][dest] = row['class_id']
            values['history_xy'][dest] = xy; values['history_boxes'][dest] = boxes
            values['history_valid'][dest] = valid; values['velocity_valid'][dest] = vv
            values['velocity_causal_fd'][dest] = velocity
            values['target_eligible'][dest] = valid.all()
            values['baseline_cv'][dest] = xy[-1]+np.arange(1,source.FUTURE+1)[:,None]*source.STRIDE*velocity[-1]
    if set(values) != set(INPUT_FIELDS):
        raise AssertionError('Exact existing causal input schema required')
    return values


def input_statistics(values):
    if set(values) != set(INPUT_FIELDS):
        raise ValueError('No future target or validity fields admitted')
    eligible = values['target_eligible']; offsets = values['query_offsets']
    target_counts = np.add.reduceat(np.r_[eligible.astype(np.int64),0],offsets[:-1]) if len(offsets)>1 else np.array([],int)
    # Empty query segments need explicit zero, rather than reduceat's next element.
    target_counts = np.where(np.diff(offsets)>0,target_counts,0)
    speed = np.linalg.norm(values['velocity_causal_fd'][eligible,-1],axis=1)
    width = np.maximum(values['history_boxes'][eligible,-1,2]-values['history_boxes'][eligible,-1,0],1e-6)
    quantile = lambda x: np.quantile(x,[0,.25,.5,.75,.95,1]).tolist() if len(x) else None
    return dict(queries=len(offsets)-1,visible_rows=len(eligible),target_rows=int(eligible.sum()),
        partial_history_context_rows=int((~eligible).sum()),
        queries_with_at_least_targets={str(k):int((target_counts>=k).sum()) for k in (1,2,5,10)},
        target_speed_pixel_per_raw_frame_quantiles=quantile(speed),
        target_speed_per_current_box_width_quantiles=quantile(speed/width),
        visible_history_valid_counts=np.bincount(values['history_valid'].sum(1),minlength=source.HISTORY+1).tolist(),
        future_target_arrays_constructed=0,forecast_errors_computed=0)


def load_record(directory, manifest, member, *, registration_sha256):
    """Lazy consumer that cannot load label arrays through this interface."""
    row=require_input_role(manifest,member)
    directory=Path(directory);receipt=json.loads((directory/'receipt.json').read_text())
    expected={'input_'+key for key in INPUT_FIELDS}|{'index_query_frames'}
    if (receipt.get('source_member')!=member or receipt.get('role')!='model_selection_inputs_only'
            or receipt.get('rows_sha256')!=row['rows_sha256']
            or receipt.get('registration_sha256')!=registration_sha256
            or set(receipt['arrays'])!=expected):
        raise ValueError('Exact input-only cache receipt required')
    result={}
    for field,ref in receipt['arrays'].items():
        name=field+'.npy'
        if ref['name']!=name:
            raise ValueError('No external or label file path allowed')
        path=directory/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('Input cache changed')
        value=np.load(path,allow_pickle=False,mmap_mode='r')
        if list(value.shape)!=ref['shape'] or str(value.dtype)!=ref['dtype']:
            raise ValueError('Input schema changed')
        result[field]=value
    return {key:result['input_'+key] for key in INPUT_FIELDS},result['index_query_frames']
