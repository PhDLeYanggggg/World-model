"""Join sampled SDD geometry and past pixels by explicit identities, never labels."""
from __future__ import annotations

import numpy as np

from src.world_model.m3w_offline_visual_forecast import geometry_features
from src.world_model.m3w_sdd_past_images import SDDPastImageStore


def visual_model_arrays(inputs, images):
    frames = np.asarray(images['source_frames'])
    expected = images['query_frame'] + inputs['history_frame_offsets']
    if (frames.shape != (8,) or not np.array_equal(frames, expected)
            or np.any(frames > images['query_frame'])):
        raise ValueError('Image and geometry past grids differ')
    if not np.array_equal(images['state_mask'], inputs['history_mask']):
        raise ValueError('Image and geometry observation masks differ')
    rgb = images['rgb_retained'].astype(np.float32)/255.
    area = images['block_area']
    support = images['retained_count'].astype(np.float32)[:, None]/area
    if rgb.shape != (8, 3, 32, 32) or support.shape != (8, 1, 32, 32):
        raise ValueError('Expected registered eight-frame image inputs')
    if (not np.isfinite(rgb).all() or not np.isfinite(support).all()
            or np.any(support < 0) or np.any(support > 1)):
        raise ValueError('Invalid observed image support')
    return dict(geometry=geometry_features(inputs), rgb=rgb, coverage=support,
                baseline=inputs['baseline_rollouts'][1].copy())


class SDDStepImageStore(SDDPastImageStore):
    """Sparse frame cache: an uncached request is an error, not an absent image."""

    def inputs(self, *, query_frame, agent_id, length=8, step=1):
        if length != 8:
            raise ValueError('This bridge is registered only for eight observed steps')
        result = super().inputs(query_frame=query_frame, agent_id=agent_id,
                                length=length, step=step)
        if not result['annotation_present_mask'].all():
            raise ValueError('Uncached past request; do not substitute absent-image masks')
        result['block_area'] = self.block_area
        return result

    def model_inputs(self, adapter, item):
        if self.metadata['source']['annotation_key'] != adapter.metadata['id']:
            raise ValueError('Geometry/image recording identity mismatch')
        identity = adapter.identity(item)
        images = self.inputs(query_frame=identity['frame_id'], agent_id=identity['agent_id'],
                             step=adapter.stride)
        provenance = adapter.get_past_provenance(item)
        if (not np.array_equal(provenance['annotation_boxes'], images['annotation_boxes'])
                or not np.array_equal(provenance['generated'], images['generated_mask'])
                or not np.array_equal(provenance['occluded'], images['occluded_mask'])):
            raise ValueError('Past annotation lineage mismatch')
        return visual_model_arrays(adapter.get_inputs(item), images)
