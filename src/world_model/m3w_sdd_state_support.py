"""Descriptive state-change support; supervised event labels never enter inputs."""
from __future__ import annotations

import numpy as np


EVENTS = ('exact_static_to_movement', 'near_static_to_movement', 'moving_to_stop',
          'turn_45_to_135', 'reverse_at_least_135')


def validate_track(track):
    a = np.asarray(track, dtype=float)
    if (a.ndim != 2 or a.shape[1] != 9 or not len(a) or not np.isfinite(a).all()
            or np.any(a[:, 3:5] < a[:, 1:3]) or len(np.unique(a[:, 0])) != 1
            or np.any(np.diff(a[:, 5]) <= 0) or np.any(a[:, [0, 5]] < 0)
            or np.any(a[:, [0, 5]] != np.rint(a[:, [0, 5]]))
            or not np.isin(a[:, 6:9], [0, 1]).all()):
        raise ValueError('One finite sorted unique-frame source track required')
    return a


def frame_lookup(track, wanted):
    frames = track[:, 5].astype(np.int64)
    wanted = np.asarray(wanted, np.int64)
    index = np.searchsorted(frames, wanted)
    safe = np.minimum(index, len(frames)-1)
    return np.where((index < len(frames)) & (frames[safe] == wanted), index, -1)


def past_window_indices(track, stride):
    """Eligibility depends on eight non-lost observed states, not future existence."""
    a = validate_track(track)
    if not isinstance(stride, (int, np.integer)) or isinstance(stride, bool) or stride <= 0:
        raise ValueError('Positive integer raw-frame stride required')
    queries = a[(a[:, 6] == 0) & (a[:, 5] % stride == 0), 5].astype(np.int64)
    indices = frame_lookup(a, queries[:, None]-np.arange(7, -1, -1)*stride)
    valid = np.all(indices >= 0, axis=1)
    safe = np.maximum(indices, 0)
    valid &= np.all(a[safe, 6] == 0, axis=1)
    return queries[valid], indices[valid]


def past_features(track, indices):
    history = track[indices]
    xy = (history[:, :, 1:3]+history[:, :, 3:5])/2
    diagonal = np.linalg.norm(history[:, :, 3:5]-history[:, :, 1:3], axis=2)
    scale = np.median(diagonal, axis=1)
    return dict(history_xy=xy, past_box_diagonal_scale=scale,
                past_path=np.linalg.norm(np.diff(xy, axis=1), axis=2).sum(axis=1),
                exact_static=np.all(xy == xy[:, -1:, :], axis=(1, 2)))


def future_event_labels(track, queries, history_indices, stride):
    """Retrospective labels for support auditing only, separate from past features."""
    past = past_features(track, history_indices)
    future_indices = frame_lookup(track, queries[:, None]+np.arange(1, 13)*stride)
    safe = np.maximum(future_indices, 0)
    valid = np.all(future_indices >= 0, axis=1) & np.all(track[safe, 6] == 0, axis=1)
    scale = past['past_box_diagonal_scale']
    valid_scale = scale > 0
    safe_scale = np.where(valid_scale, scale, 1)
    future = (track[safe, 1:3]+track[safe, 3:5])/2
    last = past['history_xy'][:, -1]
    joined = np.concatenate((last[:, None], future), axis=1)
    future_path = np.linalg.norm(np.diff(joined, axis=1), axis=2).sum(axis=1)
    max_travel = np.linalg.norm(future-last[:, None], axis=2).max(axis=1)
    late_path = np.linalg.norm(np.diff(future[:, -4:], axis=1), axis=2).sum(axis=1)
    past_direction = past['history_xy'][:, -1]-past['history_xy'][:, -4]
    future_direction = future[:, -1]-future[:, -4]
    denom = np.linalg.norm(past_direction, axis=1)*np.linalg.norm(future_direction, axis=1)
    cosine = np.divide(np.sum(past_direction*future_direction, axis=1), denom,
                       out=np.ones_like(denom), where=denom > 0)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    defined = valid & valid_scale
    moving = past['past_path']/safe_scale >= .5
    continued = (future_path/safe_scale >= .5) & (denom > 0)
    events = dict(
        exact_static_to_movement=defined & past['exact_static'] & (max_travel/safe_scale >= .5),
        near_static_to_movement=defined & (past['past_path']/safe_scale <= .1) & (max_travel/safe_scale >= .5),
        moving_to_stop=defined & moving & (late_path/safe_scale <= .05),
        turn_45_to_135=defined & moving & continued & (angle >= 45) & (angle < 135),
        reverse_at_least_135=defined & moving & continued & (angle >= 135))
    unoccluded = np.all(track[history_indices, 7] == 0, axis=1) & np.all(track[safe, 7] == 0, axis=1)
    return dict(events=events, future_complete_nonlost=valid, past_scale_valid=valid_scale,
                all_twenty_unoccluded=valid & unoccluded, future_indices=future_indices)


def control_provenance(track):
    a = validate_track(track)
    frames = a[:, 5].astype(np.int64)
    controls = np.flatnonzero(a[:, 8] == 0)
    previous = np.full(len(a), -1, np.int64)
    following = previous.copy()
    error = np.full(len(a), np.nan)
    if len(controls):
        control_frames = frames[controls]
        left = np.searchsorted(control_frames, frames, side='right')-1
        right = np.searchsorted(control_frames, frames, side='left')
        previous[left >= 0] = controls[left[left >= 0]]
        following[right < len(controls)] = controls[right[right < len(controls)]]
        bracketed = (previous >= 0) & (following >= 0)
        ids = np.flatnonzero(bracketed)
        before, after = previous[ids], following[ids]
        span = frames[after]-frames[before]
        fraction = np.divide(frames[ids]-frames[before], span,
                             out=np.zeros(len(ids)), where=span > 0)
        interpolated = a[before, 1:5]+fraction[:, None]*(a[after, 1:5]-a[before, 1:5])
        error[ids] = np.abs(a[ids, 1:5]-interpolated).max(axis=1)
    else:
        bracketed = np.zeros(len(a), bool)
    return dict(previous_control_row=previous, next_control_row=following,
                bracketed=bracketed, max_box_interpolation_error=error,
                control_gaps=np.diff(frames[controls]))


def nonoverlap_count(queries, stride):
    """Greedy disjoint 8+12 raw-frame spans; disjoint is not statistically IID."""
    end, count = -1, 0
    for q in np.sort(np.asarray(queries, np.int64)):
        if q-7*stride > end:
            count += 1
            end = int(q+12*stride)
    return count


def track_support(track, stride, provenance=None):
    a = validate_track(track)
    p = control_provenance(a) if provenance is None else provenance
    queries, histories = past_window_indices(a, stride)
    output = dict(past_eligible=len(queries), future_complete_nonlost=0,
        past_scale_invalid=0, histories_with_post_query_controls=0,
        histories_with_unbracketed_generated_rows=0, histories_with_at_least_two_controls=0,
        past_eligible_tracks=int(bool(len(queries))))
    for event in EVENTS:
        output.update({event+'_windows':0, event+'_tracks':0, event+'_disjoint_spans':0,
                       event+'_unoccluded_windows':0})
    if not len(queries):
        return output
    labels = future_event_labels(a, queries, histories, stride)
    generated = a[histories, 8] == 1
    next_rows = p['next_control_row'][histories]
    next_frames = a[np.maximum(next_rows, 0), 5]
    later = generated & (next_rows >= 0) & (next_frames > queries[:, None])
    output.update(future_complete_nonlost=int(labels['future_complete_nonlost'].sum()),
        past_scale_invalid=int((~labels['past_scale_valid']).sum()),
        histories_with_post_query_controls=int(np.any(later, axis=1).sum()),
        histories_with_unbracketed_generated_rows=int(np.any(generated & ~p['bracketed'][histories], axis=1).sum()),
        histories_with_at_least_two_controls=int((np.sum(~generated, axis=1) >= 2).sum()))
    for event, mask in labels['events'].items():
        output.update({event+'_windows':int(mask.sum()), event+'_tracks':int(mask.any()),
            event+'_disjoint_spans':nonoverlap_count(queries[mask], stride),
            event+'_unoccluded_windows':int((mask & labels['all_twenty_unoccluded']).sum())})
    return output
