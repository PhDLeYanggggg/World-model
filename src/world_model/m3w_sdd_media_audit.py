"""Dataset-local SDD media checks, without assigning training/evaluation roles."""
from __future__ import annotations

import numpy as np


def annotation_summary(rows, width, height):
    rows = np.asarray(rows, dtype=float)
    if (rows.ndim != 2 or rows.shape[1] != 9 or not len(rows)
            or not np.isfinite(rows).all() or width <= 0 or height <= 0):
        raise ValueError('Nonempty finite nine-column annotation geometry required')
    integer = rows[:, [0, 5, 6, 7, 8]]
    if (np.any(integer != np.rint(integer)) or np.any(integer[:, :2] < 0)
            or not np.isin(integer[:, 2:], [0, 1]).all()):
        raise ValueError('Invalid source IDs, frame indices or annotation flags')
    if np.any(rows[:, 3:5] < rows[:, 1:3]):
        raise ValueError('Reversed annotation boxes')
    keys = rows[:, [0, 5]].astype(np.int64)
    frames, counts = np.unique(keys[:, 1], return_counts=True)
    visible = rows[:, 6] == 0
    in_bounds = (rows[:, 1] >= 0) & (rows[:, 2] >= 0) & (rows[:, 3] <= width) & (rows[:, 4] <= height)
    return dict(rows=len(rows), tracks=len(np.unique(keys[:, 0])),
        frame_min=int(frames[0]), frame_max=int(frames[-1]), unique_frames=len(frames),
        duplicate_agent_frame_rows=int(len(keys)-len(np.unique(keys, axis=0))),
        frame_union_contiguous=bool(np.all(np.diff(frames) == 1)),
        visible_rows=int(visible.sum()), lost_rows=int((~visible).sum()),
        occluded_rows=int(rows[:, 7].sum()), generated_rows=int(rows[:, 8].sum()),
        visible_boxes_outside_image=int((visible & ~in_bounds).sum()),
        rows_per_frame_quantiles=np.quantile(counts, [0, .5, .9, 1]).tolist())


def decoded_coverage(annotation_frames, decoded_frames, header_frames):
    frames = np.asarray(annotation_frames)
    if (frames.ndim != 1 or not len(frames) or not np.isfinite(frames).all()
            or np.any(frames != np.rint(frames)) or np.any(frames < 0)
            or decoded_frames < 0 or header_frames < 0):
        raise ValueError('Invalid annotation or decoded frame count')
    covered = frames < decoded_frames
    return dict(decoded_frames=int(decoded_frames), header_frames=int(header_frames),
        header_matches_decode=bool(decoded_frames == header_frames),
        annotation_rows_with_decodable_index=int(covered.sum()),
        annotation_rows_outside_decode=int((~covered).sum()),
        decoded_range_covers_annotations=bool(covered.all()),
        decoded_frames_without_annotation_index=int(max(0, decoded_frames-len(np.unique(frames[covered])))),
        semantic_image_annotation_alignment_certified=False,
        physical_time_or_scale_certified=False)


def reference_comparison(frame, reference):
    a, b = np.asarray(frame), np.asarray(reference)
    if a.shape != b.shape:
        return dict(shape_match=False, frame_shape=list(a.shape), reference_shape=list(b.shape),
                    semantic_alignment_certified=False)
    delta = a.astype(float)-b.astype(float)
    flat_a, flat_b = a.astype(float).ravel(), b.astype(float).ravel()
    return dict(shape_match=True, mean_absolute_pixel_difference=float(np.abs(delta).mean()),
        pixel_correlation=float(np.corrcoef(flat_a, flat_b)[0, 1])
            if flat_a.std() and flat_b.std() else None,
        exact_pixels=bool(np.array_equal(a, b)), semantic_alignment_certified=False)
