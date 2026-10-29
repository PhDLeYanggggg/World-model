"""Unsmoothed observed boxes/confidences; no target or future-support inputs."""
import numpy as np

GROUPS=('log_width','log_height','confidence','log_aspect','half_height_step',
        'center_dx_body_scale','center_dy_body_scale','box_iou')
NAMES=tuple(f'{name}_past{j}' for name in GROUPS for j in range(8))
REPRESENTATIONS=('disabled','current','noise','history')


def lookup_history(rows,agents,query_frames,expected_boxes):
    agents,query_frames=np.asarray(agents),np.asarray(query_frames)
    n=len(agents)
    if (agents.shape!=(n,) or query_frames.shape!=(n,) or expected_boxes.shape!=(n,8,4)
            or not np.issubdtype(agents.dtype,np.integer) or not np.issubdtype(query_frames.dtype,np.integer)):
        raise ValueError('Integer query identities and complete eight-box histories required')
    dtype=np.dtype([('agent','<i8'),('frame','<i8')])
    raw=np.empty(len(rows),dtype=dtype);raw['agent']=rows['agent'];raw['frame']=rows['frame']
    order=np.argsort(raw,kind='stable');ordered=raw[order]
    if np.any(ordered[1:]==ordered[:-1]):
        raise ValueError('Repeated raw agent/frame keys')
    frames=query_frames[:,None]-12*np.arange(7,-1,-1)[None]
    keys=np.empty(n*8,dtype=dtype);keys['agent']=np.repeat(agents,8);keys['frame']=frames.reshape(-1)
    pos=np.searchsorted(ordered,keys)
    valid=pos<len(ordered)
    valid[valid]&=ordered[pos[valid]]==keys[valid]
    if not valid.all() or np.any(frames>query_frames[:,None]):
        raise ValueError('Every observation must map exactly to the current or past raw frame')
    take=order[pos].reshape(n,8)
    boxes=np.stack([rows[k][take] for k in ('x_min','y_min','x_max','y_max')],-1)
    np.testing.assert_array_equal(boxes,expected_boxes)
    confidence=rows['confidence'][take].astype(np.float64)
    return boxes,confidence,frames


def features(boxes,confidence,representation='history'):
    boxes,confidence=map(np.asarray,(boxes,confidence))
    if (boxes.ndim!=3 or boxes.shape[1:]!=(8,4) or confidence.shape!=boxes.shape[:2]
            or not np.isfinite(boxes).all() or not np.isfinite(confidence).all()
            or np.any(boxes[...,2:]<boxes[...,:2]) or (confidence<0).any() or (confidence>1).any()
            or representation not in ('history','current','disabled')):
        raise ValueError('Finite ordered past boxes and confidence in[0,1] required')
    if representation=='disabled':
        return np.zeros((len(boxes),64),np.float32)
    if representation=='current':
        boxes=np.repeat(boxes[:,-1:],8,axis=1);confidence=np.repeat(confidence[:,-1:],8,axis=1)
    size=boxes[...,2:]-boxes[...,:2];center=(boxes[...,:2]+boxes[...,2:])/2
    width,height=size[...,0],size[...,1];body=1+np.median(height,axis=1)
    logw,logh=np.log1p(width),np.log1p(height)
    dh=np.zeros_like(height);dh[:,1:]=np.diff(height,axis=1)/(2*body[:,None])
    step=np.zeros_like(center);step[:,1:]=np.diff(center,axis=1)/body[:,None,None]
    lo=np.maximum(boxes[:,:-1,:2],boxes[:,1:,:2]);hi=np.minimum(boxes[:,:-1,2:],boxes[:,1:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    area=size.prod(-1);union=area[:,:-1]+area[:,1:]-intersection
    iou=np.zeros_like(height);iou[:,1:]=np.divide(intersection,union,out=np.zeros_like(union),where=union>0)
    out=np.concatenate((logw,logh,confidence,logw-logh,dh,step[...,0],step[...,1],iou),axis=1).astype(np.float32)
    if out.shape!=(len(boxes),64) or not np.isfinite(out).all():
        raise AssertionError('Fixed finite64-field schema changed')
    return out


def noise_features(recordings,agents,frames):
    """A fixed-capacity control determined only by currently available identities."""
    recordings,agents,frames=[np.asarray(v,dtype=np.uint64) for v in (recordings,agents,frames)]
    if recordings.ndim!=1 or agents.shape!=recordings.shape or frames.shape!=recordings.shape:
        raise ValueError('Aligned causal query keys required')
    out=np.empty((len(frames),64),np.float32)
    with np.errstate(over='ignore'):
        for start in range(0,len(frames),2048):
            sl=slice(start,start+2048)
            key=recordings[sl]*np.uint64(0x9e3779b97f4a7c15)^agents[sl]*np.uint64(0xbf58476d1ce4e5b9)^frames[sl]
            x=key[:,None]+np.arange(64,dtype=np.uint64)[None]*np.uint64(0x94d049bb133111eb)+np.uint64(20261029)
            x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            x=x^(x>>np.uint64(31))
            out[sl]=(((x>>np.uint64(11)).astype(np.float64)/2**53)*2-1).astype(np.float32)
    return out
