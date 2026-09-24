"""Labels-only HT21 structural audit. No forecasting or scientific admission."""
import configparser
import io

import numpy as np

CLASS_NAMES = {1:'pedestrian', 2:'static', 3:'ignore', 4:'person_on_vehicle'}


def read_metadata(payload):
    config = configparser.ConfigParser()
    config.read_string(payload.decode('utf-8'))
    s = config['Sequence']
    result = dict(name=s['name'], frames=s.getint('seqLength'), fps=s.getfloat('frameRate'),
                  width=s.getint('imWidth'),height=s.getint('imHeight'),
                  camera_motion=s.getboolean('cam_motion', fallback=None))
    if min(result[k] for k in ('frames','fps','width','height')) <= 0:
        raise ValueError('Positive declared dimensions and sampling rate required')
    return result


def read_ground_truth(payload, metadata):
    rows = np.loadtxt(io.BytesIO(payload),delimiter=',',ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 9 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('Finite nine-column ground truth required')
    if not np.array_equal(rows[:,[0,1,6,7]],np.floor(rows[:,[0,1,6,7]])):
        raise ValueError('Frame, identity, confidence mark and class must be integer-valued')
    if ((rows[:,0]<1).any() or (rows[:,0]>metadata['frames']).any() or (rows[:,1]<0).any()
            or (rows[:,4:6]<=0).any() or not np.isin(rows[:,6],[0,1]).all()
            or not np.isin(rows[:,7],list(CLASS_NAMES)).all()
            or (rows[:,8]<0).any() or (rows[:,8]>1).any()):
        raise ValueError('Invalid frame range, box geometry, mark, class or visibility')
    order = np.lexsort((rows[:,0],rows[:,1]))
    sorted_rows = rows[order]
    if np.any(np.all(sorted_rows[1:,:2] == sorted_rows[:-1,:2],axis=1)):
        raise ValueError('Duplicate frame/agent ground truth')
    return rows


def human_observation_mask(rows):
    # Static and pedestrian labels are merged, not used to predict future motion.
    return np.isin(rows[:,7],[1,2,4]) & (rows[:,6]>0) & (rows[:,8]>0)


def history(rows, agent_id, query_frame, *, length=8, stride=1):
    if length < 2 or stride < 1:
        raise ValueError('Positive stride and at least two history steps required')
    requested = query_frame - np.arange(length-1,-1,-1)*stride
    use = ((rows[:,1] == agent_id) & (rows[:,0] <= query_frame) & human_observation_mask(rows))
    prefix = rows[use]
    order = np.argsort(prefix[:,0]); prefix = prefix[order]
    where = np.searchsorted(prefix[:,0],requested)
    valid = where < len(prefix)
    valid[valid] &= prefix[where[valid],0] == requested[valid]
    xy = np.zeros((length,2))
    xy[valid] = prefix[where[valid],2:4]+.5*prefix[where[valid],4:6]
    velocity = np.zeros_like(xy)
    velocity_valid = np.zeros(length,bool)
    velocity_valid[1:] = valid[1:] & valid[:-1]
    velocity[1:] = np.where(velocity_valid[1:,None],np.diff(xy,axis=0)/stride,0.)
    return dict(frame_id=requested,xy=xy,valid_mask=valid,velocity_causal_fd=velocity,
                velocity_valid_mask=velocity_valid,agent_type='person',
                coordinate_unit='head_center_annotation_pixel',
                provenance='past_indexed_offline_annotations_not_verified_online_tracking')


def counts(values):
    keys,n = np.unique(values.astype(int),return_counts=True)
    return {str(k):int(v) for k,v in zip(keys,n)}


def availability(rows, frames, *, lengths=(8,16,32,64)):
    """Eligibility uses history only; future support is a separate audit count."""
    valid_rows = rows[human_observation_mask(rows)]
    order = np.lexsort((valid_rows[:,0],valid_rows[:,1]))
    r = valid_rows[order]
    parts = np.split(r,np.flatnonzero(np.diff(r[:,1]))+1) if len(r) else []
    result = {str(k):dict(past_eligible=0,complete_future12=0,partial_future12=0,
                         no_future12=0,endpoint_available={str(h):0 for h in (10,25,50,100)})
              for k in lengths}
    per_frame = {str(k):np.zeros(frames+1,dtype=np.int64) for k in lengths}
    for track in parts:
        f = track[:,0].astype(int)
        for length in lengths:
            past = np.isin(f[:,None]-np.arange(length),f).all(1)
            current = f[past]
            per_frame[str(length)] += np.bincount(current,minlength=frames+1)
            future = np.isin(current[:,None]+np.arange(1,13),f)
            out = result[str(length)]
            out['past_eligible'] += int(past.sum())
            out['complete_future12'] += int(future.all(1).sum())
            out['no_future12'] += int((~future.any(1)).sum())
            out['partial_future12'] += int((future.any(1) & ~future.all(1)).sum())
            for h in (10,25,50,100):
                out['endpoint_available'][str(h)] += int(np.isin(current+h,f).sum())
    for k,c in per_frame.items():
        result[k]['query_frames_with_history'] = int((c>0).sum())
        result[k]['query_frames_at_least_agents'] = {str(n):int((c>=n).sum()) for n in (2,5,10,50,100)}
        result[k]['eligible_agents_per_nonempty_frame'] = (
            dict(zip(('min','median','p95','max'),map(float,np.percentile(c[c>0],[0,50,95,100]))))
            if (c>0).any() else None)
    return result


def audit_ground_truth(rows, metadata):
    order = np.lexsort((rows[:,0],rows[:,1]))
    r = rows[order]
    parts = np.split(r,np.flatnonzero(np.diff(r[:,1]))+1)
    spans, observations, gaps, class_changes, second_diff, triples = [],[],0,0,0,0
    for t in parts:
        f = t[:,0].astype(int)
        spans.append(int(f[-1]-f[0]+1)); observations.append(len(t))
        gaps += int((np.diff(f)>1).sum())
        class_changes += int(len(np.unique(t[:,7]))>1)
        if len(t)>=3:
            eligible = (np.diff(f)[:-1]==1) & (np.diff(f)[1:]==1)
            xy = t[:,2:4]+.5*t[:,4:6]
            d2 = np.diff(xy,n=2,axis=0)
            triples += int(eligible.sum())
            second_diff += int((eligible & (np.max(np.abs(d2),axis=1)<=1e-8)).sum())
    visible = human_observation_mask(rows)
    frame_counts = np.bincount(rows[visible,0].astype(int),minlength=metadata['frames']+1)[1:]
    stat = lambda a:dict(zip(('min','median','p95','max'),map(float,np.percentile(a,[0,50,95,100]))))
    return dict(rows=len(rows),track_ids=len(parts),row_class_counts=counts(rows[:,7]),
        confidence_mark_counts=counts(rows[:,6]),visibility_counts=counts(rows[:,8]*1000),
        merged_visible_human_rows=int(visible.sum()),
        visible_static_rows=int((visible & (rows[:,7]==2)).sum()),
        classes_are_audit_metadata_not_motion_features=True,
        track_row_count=stat(observations),track_frame_span=stat(spans),
        nonconsecutive_track_edges=gaps,tracks_with_class_changes=class_changes,
        visible_humans_per_frame=stat(frame_counts),
        consecutive_triples=triples,exact_linear_center_triples=second_diff,
        exact_linear_fraction=second_diff/triples if triples else None,
        linearity_is_not_proof_of_keyframe_or_interpolation_origin=True,
        out_of_image_box_rows=int(((rows[:,2]<0)|(rows[:,3]<0)|
            (rows[:,2]+rows[:,4]>metadata['width'])|(rows[:,3]+rows[:,5]>metadata['height'])).sum()),
        availability_stride1=availability(rows,metadata['frames']))
