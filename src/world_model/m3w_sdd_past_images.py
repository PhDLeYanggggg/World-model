"""Offline annotated past images with separate geometric and suspect-padding support."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.evaluation.m3w_experiment_contract import file_digest


ARRAYS = ('row_keys', 'annotation_boxes', 'image_boxes', 'source_flags',
          'rgb_observed', 'rgb_retained', 'geometric_count', 'retained_count')


def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError('Invalid ' + name)
    return int(value)


def border_padding_suspect(image, threshold=8):
    """Four-connected, near-black image-edge components; an inference, not a label."""
    image = np.asarray(image)
    threshold = _integer(threshold, 'black threshold')
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or not image.shape[0] or not image.shape[1] or threshold > 255):
        raise ValueError('Nonempty uint8 RGB and a byte threshold required')
    dark = np.max(image, axis=2) <= threshold
    components, _ = ndimage.label(dark, structure=ndimage.generate_binary_structure(2, 1))
    border_ids = np.unique(np.concatenate((components[0], components[-1],
                                          components[:, 0], components[:, -1])))
    border_ids = border_ids[border_ids != 0]
    return np.isin(components, border_ids) if len(border_ids) else np.zeros(dark.shape, bool)


def supported_center_patch(image, center, suspect, crop_size=96, output_size=32):
    """Keep unmodified observed RGB and a second view excluding suspect border pixels."""
    image, center, suspect = np.asarray(image), np.asarray(center, float), np.asarray(suspect)
    crop_size = _integer(crop_size, 'crop size', 1)
    output_size = _integer(output_size, 'output size', 1)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or suspect.shape != image.shape[:2] or suspect.dtype != np.bool_
            or center.shape != (2,) or not np.isfinite(center).all()
            or crop_size % output_size or (crop_size // output_size)**2 > 255):
        raise ValueError('Invalid image, center, mask or pooling dimensions')
    left, top = [int(v) - crop_size//2 for v in np.rint(center)]
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left+crop_size, image.shape[1]), min(top+crop_size, image.shape[0])
    patch = np.zeros((crop_size, crop_size, 3), np.uint8)
    geometric = np.zeros((crop_size, crop_size), bool)
    retained = geometric.copy()
    if x1 > x0 and y1 > y0:
        ys, xs = slice(y0-top, y1-top), slice(x0-left, x1-left)
        patch[ys, xs] = image[y0:y1, x0:x1]
        geometric[ys, xs] = True
        retained[ys, xs] = ~suspect[y0:y1, x0:x1]
    block = crop_size//output_size
    result = {}
    for rgb_name, count_name, mask in (
            ('rgb_observed', 'geometric_count', geometric),
            ('rgb_retained', 'retained_count', retained)):
        count = mask.reshape(output_size, block, output_size, block).sum((1, 3), dtype=np.uint16)
        values = (patch*mask[..., None]).reshape(output_size, block, output_size, block, 3)
        sums = values.sum((1, 3), dtype=np.uint32)
        mean = np.zeros_like(sums, dtype=float)
        np.divide(sums, count[..., None], out=mean, where=count[..., None] > 0)
        result[rgb_name] = np.rint(mean).astype(np.uint8).transpose(2, 0, 1)
        result[count_name] = count.astype(np.uint8)
    return result


def validate_arrays(arrays, *, output_size, block_area, decoded_prefix_frames):
    if set(arrays) != set(ARRAYS):
        raise ValueError('Unexpected input cache schema; labels are not accepted')
    keys = arrays['row_keys']
    n = len(keys)
    if (keys.shape != (n, 2) or not np.issubdtype(keys.dtype, np.integer)
            or np.any(keys < 0) or np.any(keys[:, 0] >= decoded_prefix_frames)
            or len(np.unique(keys, axis=0)) != n):
        raise ValueError('Invalid or duplicate frame/agent keys')
    for name in ('annotation_boxes', 'image_boxes'):
        boxes = arrays[name]
        if boxes.shape != (n, 4) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError('Invalid boxes')
    flags = arrays['source_flags']
    if flags.shape != (n, 3) or flags.dtype != np.uint8 or not np.isin(flags, (0, 1)).all():
        raise ValueError('Invalid lost/occluded/generated flags')
    for rgb, count in (('rgb_observed', 'geometric_count'), ('rgb_retained', 'retained_count')):
        if (arrays[rgb].shape != (n, 3, output_size, output_size) or arrays[rgb].dtype != np.uint8
                or arrays[count].shape != (n, output_size, output_size) or arrays[count].dtype != np.uint8
                or np.any(arrays[count] > block_area)
                or np.any(arrays[rgb]*(arrays[count][:, None] == 0))):
            raise ValueError('Invalid pixel values or support counts')
    if (np.any(arrays['retained_count'] > arrays['geometric_count'])
            or np.any(arrays['geometric_count'][flags[:, 0] == 1])):
        raise ValueError('Suspect filtering cannot add support; lost boxes cannot supply pixels')


class SDDPastImageStore:
    """Hash-checked diagnostic store; no training admission or sensor-as-of claim."""

    def __init__(self, directory, *, observation_mode, data_role='diagnostic_only'):
        if data_role != 'diagnostic_only' or observation_mode != 'offline_annotated':
            raise ValueError('Only diagnostic offline-annotated access is admitted')
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory/'metadata.json').read_text())
        m = self.metadata
        if (m['data_role'] != 'diagnostic_only' or m['training_admitted']
                or m['observation_mode'] != 'offline_annotated'
                or set(m['array_sha256']) != set(ARRAYS)):
            raise ValueError('Cache admission or schema mismatch')
        self.output_size = _integer(m['output_size'], 'output size', 1)
        crop = _integer(m['crop_size'], 'crop size', 1)
        self.prefix = _integer(m['decoded_prefix_frames'], 'decoded prefix', 1)
        if crop % self.output_size or (crop//self.output_size)**2 > 255:
            raise ValueError('Invalid pooling')
        self.block_area = (crop//self.output_size)**2
        self.arrays = {}
        for name in ARRAYS:
            path = self.directory/(name+'.npy')
            if file_digest(path) != m['array_sha256'][name]:
                raise ValueError('Changed image input array: '+name)
            self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        validate_arrays(self.arrays, output_size=self.output_size, block_area=self.block_area,
                        decoded_prefix_frames=self.prefix)
        self.lookup = {tuple(map(int, key)): i for i, key in enumerate(self.arrays['row_keys'])}

    def _query(self, query_frame):
        q = _integer(query_frame, 'query frame')
        if q >= self.prefix:
            raise ValueError('Query outside decoded prefix; no silent missing-tail substitution')
        return q

    def agents_at(self, query_frame):
        q = self._query(query_frame)
        keys, flags = self.arrays['row_keys'], self.arrays['source_flags']
        return np.unique(keys[(keys[:, 0] <= q) & (flags[:, 0] == 0), 1])

    def inputs(self, *, query_frame, agent_id, length=8, step=1):
        q, agent = self._query(query_frame), _integer(agent_id, 'agent id')
        length, step = _integer(length, 'history length', 1), _integer(step, 'history step', 1)
        if length not in (8, 16, 32, 64):
            raise ValueError('Supported diagnostic history lengths: 8/16/32/64')
        if agent not in self.agents_at(q):
            raise ValueError('Agent not observed at or before query')
        frames = q-np.arange(length-1, -1, -1)*step
        indices = np.array([self.lookup.get((int(f), agent), -1) for f in frames])
        present = indices >= 0
        result = {'source_frames': frames, 'annotation_present_mask': present,
                  'query_frame': q, 'agent_id': agent}
        flags = np.zeros((length, 3), np.uint8)
        flags[present] = self.arrays['source_flags'][indices[present]]
        result['lost_mask'], result['occluded_mask'], result['generated_mask'] = flags.T.astype(bool)
        state_mask = present & ~result['lost_mask']
        result['state_mask'] = state_mask
        result['agent_unoccluded_flag'] = state_mask & ~result['occluded_mask']
        for name in ('rgb_observed', 'rgb_retained', 'geometric_count', 'retained_count'):
            array = self.arrays[name]
            out = np.zeros((length,)+array.shape[1:], array.dtype)
            out[present] = array[indices[present]]
            result[name] = out
        boxes = np.zeros((length, 4), float)
        boxes[state_mask] = self.arrays['annotation_boxes'][indices[state_mask]]
        result['annotation_boxes'] = boxes
        result['image_frame_mask'] = np.any(result['retained_count'] > 0, axis=(1, 2))
        return result
