"""Partial past-image support, without admitting a new scientific data role."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest


ARRAYS = ('rgb', 'coverage', 'native_xy', 'image_xy', 'row_keys', 'source_frame',
          'latest_control_frame', 'history_rows', 'frame_decode_mask')


def masked_center_patch(image, image_xy, crop_size=96, output_size=32):
    """Mean visible pixels per block; coverage counts distinguish padding from black."""
    image, xy = np.asarray(image), np.asarray(image_xy, dtype=float)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or xy.shape != (2,) or not np.isfinite(xy).all()
            or crop_size < 1 or output_size < 1 or crop_size % output_size):
        raise ValueError('uint8 RGB, finite xy and exactly divisible positive crop sizes required')
    block = crop_size // output_size
    if block * block > 255:
        raise ValueError('Coverage count must fit uint8')
    center_x, center_y = (int(x) for x in np.rint(xy))
    left, top = center_x - crop_size // 2, center_y - crop_size // 2
    right, bottom = left + crop_size, top + crop_size
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(right, image.shape[1]), min(bottom, image.shape[0])
    patch = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        patch[y0-top:y1-top, x0-left:x1-left] = image[y0:y1, x0:x1]
        mask[y0-top:y1-top, x0-left:x1-left] = 1
    sums = patch.reshape(output_size, block, output_size, block, 3).sum((1, 3), dtype=np.uint32)
    count = mask.reshape(output_size, block, output_size, block).sum((1, 3), dtype=np.uint16)
    mean = np.zeros_like(sums, dtype=float)
    np.divide(sums, count[..., None], out=mean, where=count[..., None] != 0)
    return np.rint(mean).astype(np.uint8).transpose(2, 0, 1), count.astype(np.uint8)


def validate_history_rows(row_keys, source_frame, history_rows, *, length=8, step=10):
    keys, frames, history = np.asarray(row_keys), np.asarray(source_frame), np.asarray(history_rows)
    if (keys.ndim != 2 or keys.shape[1] != 2 or len(frames) != len(keys)
            or frames.ndim != 1 or history.ndim != 2 or history.shape[1] != length
            or not np.issubdtype(history.dtype, np.integer)
            or not np.isfinite(keys).all() or not np.isfinite(frames).all()
            or not np.array_equal(keys, np.round(keys)) or not np.array_equal(frames, np.round(frames))
            or np.any(history < 0) or np.any(history >= len(keys))):
        raise ValueError('Invalid row keys, source frames or history indices')
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('Duplicate frame/agent keys')
    if (np.any(frames < 0) or np.any(keys[:, 0] - 1 != frames)
            or np.any(np.diff(frames[history], axis=1) != step)
            or np.any(keys[history, 1] != keys[history[:, -1], 1, None])):
        raise ValueError('History must remain on one agent with exact past source indices')


class MaskedHistoryImageStore:
    """Lazy diagnostic-only input reader; targets and provenance are separate APIs."""

    def __init__(self, directory, *, observation_mode, data_role='diagnostic_only'):
        if data_role != 'diagnostic_only':
            raise ValueError('Formal training/evaluation admission is pending scientific observation decision')
        if observation_mode not in ('offline_annotation_diagnostic', 'control_as_of_query_diagnostic'):
            raise ValueError('Explicit diagnostic observation mode required')
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        if (self.metadata['data_role'] != 'diagnostic_only' or self.metadata['source_role'] != 'fit'
                or set(self.metadata['array_sha256']) != set(ARRAYS)
                or self.metadata['formal_training_admitted']):
            raise ValueError('Unexpected role, array schema or admission')
        self.arrays = {}
        for name in ARRAYS:
            path = self.directory / (name + '.npy')
            if file_digest(path) != self.metadata['array_sha256'][name]:
                raise ValueError('Changed image cache array: ' + name)
            self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        a = self.arrays
        validate_history_rows(a['row_keys'], a['source_frame'], a['history_rows'])
        n, size = len(a['row_keys']), self.metadata['output_size']
        if (a['rgb'].shape != (n, 3, size, size) or a['coverage'].shape != (n, size, size)
                or a['native_xy'].shape != (n, 2) or a['image_xy'].shape != (n, 2)
                or a['frame_decode_mask'].shape != (n,) or a['frame_decode_mask'].dtype != np.bool_
                or a['latest_control_frame'].shape != (n,)
                or np.any(a['latest_control_frame'] < a['source_frame'])
                or a['rgb'].dtype != np.uint8 or a['coverage'].dtype != np.uint8
                or not np.isfinite(a['native_xy']).all() or not np.isfinite(a['image_xy']).all()
                or self.metadata['crop_size'] % size):
            raise ValueError('Invalid input array shapes, dtypes or provenance')
        self.block_area = (self.metadata['crop_size'] // size) ** 2
        if np.any(a['coverage'] > self.block_area):
            raise ValueError('Coverage exceeds real block area')
        self.observation_mode = observation_mode

    def __len__(self):
        return len(self.arrays['history_rows'])

    def _rows(self, index):
        if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(self):
            raise IndexError('History index outside cache')
        return self.arrays['history_rows'][index]

    def provenance(self, index):
        rows = self._rows(index)
        latest = self.arrays['latest_control_frame'][rows].copy()
        query = int(self.arrays['source_frame'][rows[-1]])
        return {'latest_control_frame': latest, 'controls_at_or_before_query': bool(np.all(latest <= query)),
                'strict_online_identity_or_sensor_availability_certified': False}

    def inputs(self, index):
        rows = self._rows(index)
        if (self.observation_mode == 'control_as_of_query_diagnostic'
                and not self.provenance(index)['controls_at_or_before_query']):
            raise ValueError('Interpolation uses a control after the query; no silent row filtering')
        a = self.arrays
        coverage = a['coverage'][rows].astype(np.float32) / self.block_area
        decoded = a['frame_decode_mask'][rows]
        coverage *= decoded[:, None, None]
        rgb = a['rgb'][rows].astype(np.float32) / 255
        rgb *= (coverage[:, None] > 0)
        native = a['native_xy'][rows].copy()
        return {'rgb': rgb, 'pixel_coverage': coverage[:, None],
            'frame_mask': np.any(coverage > 0, axis=(1, 2)),
            'history_native_xy': native, 'relative_native_xy': native - native[-1],
            'source_frames': a['source_frame'][rows].copy(),
            'query_source_frame': int(a['source_frame'][rows[-1]]),
            'agent_id': int(a['row_keys'][rows[-1], 1])}
