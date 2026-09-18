"""Count annotation episodes without equating overlapping windows with events."""
import numpy as np

from src.world_model.m3w_sdd_state_support import validate_track, frame_lookup


def past_episode_context(track, queries, history_span=84):
    """Past-indexed plateau identity; post-query rows cannot alter its features."""
    queries = np.asarray(queries, dtype=np.int64)
    if queries.ndim != 1 or not len(queries) or history_span < 0:
        raise ValueError('Nonempty query vector and nonnegative span required')
    # Exclude all later rows before validation or feature computation.
    a = validate_track(np.asarray(track)[np.asarray(track)[:, 5] <= queries.max()])
    qids = frame_lookup(a, queries)
    if np.any(qids < 0) or a[qids, 6].any():
        raise ValueError('Every query must have a non-lost source state')
    frames = a[:, 5].astype(np.int64)
    xy = (a[:, 1:3]+a[:, 3:5])/2
    boundary = np.r_[True, (np.diff(frames) != 1) | (a[1:, 6] != 0) |
                     (a[:-1, 6] != 0) | np.any(xy[1:] != xy[:-1], axis=1)]
    starts = np.maximum.accumulate(np.where(boundary, np.arange(len(a)), 0))
    start_frame = frames[starts[qids]]
    count = np.zeros(len(queries), np.int64)
    controls = count.copy()
    for i, (q, j) in enumerate(zip(queries, qids)):
        begin = np.searchsorted(frames, q-history_span)
        past = a[begin:j+1]
        count[i] = np.count_nonzero(past[:, 6] == 0)
        controls[i] = np.count_nonzero((past[:, 6] == 0) & (past[:, 8] == 0))
    return dict(plateau_start_frame=start_frame, plateau_age_raw_frames=queries-start_frame,
                full_raw_history_static=queries-start_frame >= history_span,
                past_nonlost_raw_rows=count, past_nonlost_control_rows=controls)


def future_change_labels(track, queries, future_span=144):
    """Retrospective audit only; never part of an inference feature interface."""
    a = validate_track(track)
    queries = np.asarray(queries, np.int64)
    current = frame_lookup(a, queries)
    if np.any(current < 0) or a[current, 6].any():
        raise ValueError('Observed non-lost queries required')
    frames = a[:, 5].astype(np.int64)
    xy = (a[:, 1:3]+a[:, 3:5])/2
    first = np.full(len(queries), -1, np.int64)
    complete = np.zeros(len(queries), bool)
    for i, (q, row) in enumerate(zip(queries, current)):
        end = np.searchsorted(frames, q+future_span, side='right')
        available = a[row+1:end, 6] == 0
        complete[i] = (end-row-1 == future_span) and available.all()
        changed = np.flatnonzero(available & np.any(xy[row+1:end] != xy[row], axis=1))
        if len(changed):
            first[i] = frames[row+1+changed[0]]
    return dict(first_changed_raw_frame=first, future_raw_complete=complete)


def nonoverlapping_window_count(tracks, queries, past_span=84, future_span=144):
    """Maximum count of equal-duration disjoint intervals within each track."""
    tracks, queries = np.asarray(tracks), np.asarray(queries, np.int64)
    if tracks.shape != queries.shape or tracks.ndim != 1:
        raise ValueError('Aligned track identifiers and query frames required')
    count = 0
    for track in np.unique(tracks):
        end = -np.inf
        for q in np.sort(queries[tracks == track]):
            if q-past_span > end:
                count += 1
                end = q+future_span
    return count


def neighbor_support(geometry):
    """Availability of the existing eight selected past-neighbor slots."""
    g = np.asarray(geometry)
    if g.ndim != 2 or g.shape[1] != 476 or not np.isfinite(g).all():
        raise ValueError('Finite frozen geometry schema required')
    p = g[:, 38:166].reshape(-1, 8, 8, 2)
    t = g[:, 166:230].reshape(-1, 8, 8)
    mask = g[:, 230:294].reshape(-1, 8, 8)
    if not np.isin(mask, [0, 1]).all() or np.any(t[mask.astype(bool)] > 0):
        raise ValueError('Binary past-only neighbor masks required')
    mask = mask.astype(bool)
    current = mask[:, :, -1] & (t[:, :, -1] == 0)
    velocity = current & mask[:, :, -2] & (t[:, :, -2] < 0)
    moving = velocity & np.any(p[:, :, -1] != p[:, :, -2], axis=2)
    return dict(current_neighbor_count=current.sum(1),
                velocity_supported_neighbor_count=velocity.sum(1),
                moving_neighbor_count=moving.sum(1),
                full_eight_neighbor_count=(mask.all(2) & current).sum(1))


def concentration(weights, groups):
    """Descriptive cluster concentration, not an inferential effective N."""
    weights, groups = np.asarray(weights, float), np.asarray(groups)
    if weights.shape != groups.shape or weights.ndim != 1 or np.any(weights < 0):
        raise ValueError('Aligned nonnegative contributions required')
    if not len(weights) or weights.sum() == 0:
        return dict(groups=0, inverse_herfindahl=None, largest_share=None, top_five_share=None)
    _, inv = np.unique(groups, return_inverse=True)
    total = np.bincount(inv, weights=weights)
    share = np.sort(total/total.sum())[::-1]
    return dict(groups=len(total), inverse_herfindahl=float(1/(share@share)),
                largest_share=float(share[0]), top_five_share=float(share[:5].sum()))
