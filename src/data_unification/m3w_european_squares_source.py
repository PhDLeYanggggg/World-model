"""Restricted raw-box source cohort; prediction inputs and labels stay separate."""
import numpy as np

HISTORY = 8
FUTURE = 12
STRIDE = 12
INPUT_FIELDS = ('query_frame', 'agent_id', 'class_id', 'history_xy', 'history_boxes',
                'history_valid', 'velocity_causal_fd', 'velocity_valid',
                'target_eligible', 'baseline_cv', 'query_offsets')
LABEL_FIELDS = ('future_xy', 'future_valid')


def past_queries(frames):
    frames = np.asarray(frames, dtype=np.int64)
    if np.any(np.diff(frames) <= 0):
        raise ValueError('Repeated or unordered agent frames')
    result = []
    for phase in np.unique(frames % STRIDE):
        values = frames[frames % STRIDE == phase]
        ends = np.r_[0, np.flatnonzero(np.diff(values) != STRIDE)+1, len(values)]
        for start, end in zip(ends[:-1], ends[1:]):
            if end-start >= HISTORY:
                result.append(values[start+HISTORY-1:end])
    return np.sort(np.concatenate(result)) if result else np.empty(0, dtype=np.int64)


def choose_query_frames(rows, maximum=256):
    if maximum < 1:
        raise ValueError('Positive query budget required')
    agents, starts, sizes = np.unique(rows['agent'], return_index=True, return_counts=True)
    if not np.array_equal(rows['agent'], np.repeat(agents, sizes)):
        raise ValueError('Rows must be sorted by agent/frame')
    support = np.zeros(int(rows['frame'].max())+1, dtype=bool)
    eligible = 0
    for start, size in zip(starts, sizes):
        query = past_queries(rows['frame'][start:start+size])
        support[query] = True
        eligible += len(query)
    query = np.flatnonzero(support)
    indices = np.linspace(0, len(query)-1, min(maximum, len(query)), dtype=np.int64)
    return query[indices], dict(all_past_eligible_targets=eligible, all_past_supported_query_frames=len(query))


def lookup(track, grid):
    idx = np.searchsorted(track['frame'], grid)
    valid = idx < len(track)
    valid[valid] &= track['frame'][idx[valid]] == grid[valid]
    boxes = np.zeros((len(grid), 4), dtype=np.float64)
    for j, field in enumerate(('x_min', 'y_min', 'x_max', 'y_max')):
        boxes[valid, j] = track[field][idx[valid]]
    return boxes, valid


def build_recording(rows, query_frames):
    if np.any(np.diff(query_frames) <= 0):
        raise ValueError('Repeated or unordered queries')
    agents, starts, sizes = np.unique(rows['agent'], return_index=True, return_counts=True)
    if not np.array_equal(rows['agent'], np.repeat(agents, sizes)):
        raise ValueError('Rows must be sorted by agent/frame')
    tracks = {int(a): rows[start:start+size] for a, start, size in zip(agents, starts, sizes)}
    for track in tracks.values():
        if np.any(np.diff(track['frame']) <= 0):
            raise ValueError('Repeated or unordered track')
    by_frame = np.argsort(rows['frame'], kind='stable')
    frames = rows['frame'][by_frame]
    counts = np.searchsorted(frames, query_frames, side='right')-np.searchsorted(frames, query_frames)
    offsets = np.r_[0, np.cumsum(counts)]
    n = int(offsets[-1])
    inputs = dict(query_frame=np.empty(n, dtype='<i8'), agent_id=np.empty(n, dtype='<i8'),
        class_id=np.empty(n, dtype='<i8'), history_xy=np.zeros((n,HISTORY,2)),
        history_boxes=np.zeros((n,HISTORY,4)), history_valid=np.zeros((n,HISTORY),dtype=bool),
        velocity_causal_fd=np.zeros((n,HISTORY,2)), velocity_valid=np.zeros((n,HISTORY),dtype=bool),
        target_eligible=np.zeros(n,dtype=bool), baseline_cv=np.zeros((n,FUTURE,2)),
        query_offsets=offsets.astype('<i8'))
    labels = dict(future_xy=np.zeros((n,FUTURE,2)), future_valid=np.zeros((n,FUTURE),dtype=bool))
    for qi, query in enumerate(query_frames):
        current = rows[by_frame[np.searchsorted(frames,query):np.searchsorted(frames,query,side='right')]]
        current = np.sort(current, order='agent')
        for j, row in enumerate(current):
            dest = int(offsets[qi])+j
            track = tracks[int(row['agent'])]
            past_grid = query-np.arange(HISTORY-1,-1,-1)*STRIDE
            boxes, valid = lookup(track, past_grid)
            xy = (boxes[:,:2]+boxes[:,2:])/2
            vvalid = np.r_[False, valid[1:] & valid[:-1]]
            velocity = np.zeros_like(xy)
            velocity[1:] = np.where(vvalid[1:,None],np.diff(xy,axis=0)/STRIDE,0)
            inputs['query_frame'][dest] = query
            inputs['agent_id'][dest] = row['agent']
            inputs['class_id'][dest] = row['class_id']
            inputs['history_boxes'][dest] = boxes
            inputs['history_xy'][dest] = xy
            inputs['history_valid'][dest] = valid
            inputs['velocity_valid'][dest] = vvalid
            inputs['velocity_causal_fd'][dest] = velocity
            inputs['target_eligible'][dest] = valid.all()
            inputs['baseline_cv'][dest] = xy[-1]+np.arange(1,FUTURE+1)[:,None]*STRIDE*velocity[-1]
            future_boxes, future_valid = lookup(track,query+np.arange(1,FUTURE+1)*STRIDE)
            labels['future_xy'][dest] = (future_boxes[:,:2]+future_boxes[:,2:])/2
            labels['future_valid'][dest] = future_valid
    if set(inputs) != set(INPUT_FIELDS) or set(labels) != set(LABEL_FIELDS):
        raise AssertionError('Input/label schema changed')
    return inputs, labels
