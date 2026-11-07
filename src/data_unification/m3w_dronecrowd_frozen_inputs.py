"""Map the offline annotation prefix onto the frozen native forecast schema."""
import numpy as np


def adapt_past(inputs):
    if any(k.startswith(('future', 'label')) for k in inputs):
        raise ValueError('Only the observed annotation prefix can be adapted')
    frames = np.asarray(inputs['observation_frames']); h = np.asarray(inputs['positions'])
    mask = np.asarray(inputs['history_mask']); ids = np.asarray(inputs['agent_ids'])
    if (frames.shape != (8,) or (np.diff(frames) != 1).any() or frames[-1] != inputs['query_frame']
            or h.shape != (len(ids), 8, 2) or mask.shape != h.shape[:2] or mask.dtype != bool
            or not mask[:, -1].all() or len(np.unique(ids)) != len(ids) or not np.isfinite(h[mask]).all()):
        raise ValueError('Expected unique current agents and eight contiguous raw annotation steps')
    eligible = mask.all(1)
    np.testing.assert_array_equal(eligible, inputs['history_complete'])
    h = np.where(mask[..., None], h, 0.)
    velocity = np.where(mask[:, -2, None], h[:, -1]-h[:, -2], 0.)
    cv = h[:, -1, None]+np.arange(1, 13)[None, :, None]*velocity[:, None]
    return dict(history_xy=h, history_valid=mask, agent_id=ids, target_eligible=eligible, baseline_cv=cv)
