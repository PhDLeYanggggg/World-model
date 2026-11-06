"""Past-image viewpoint conditioning; no observed future pose or metric claim."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import numpy as np
import torch
from src.world_model.m3w_context_conditioning import INPUT_KEYS
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry

ARMS = ('identity', 'camera_frame')
CAMERA_KEYS = {'observed_camera_maps', 'predicted_camera_maps'}


def query_maps(flows, query_frame):
    keys = list(range(int(query_frame)-6, int(query_frame)+1))
    if query_frame < 7 or set(flows) != set(keys):
        raise ValueError('Exactly seven observed transitions ending at the query required')
    records = [flows[k] for k in keys]
    supported = np.array([r['supported'] for r in records], bool)
    past = np.broadcast_to(np.eye(3), (8, 3, 3)).copy()
    prediction = np.broadcast_to(np.eye(3), (12, 3, 3)).copy()
    if supported.all():
        affine = np.array([r['native_affine'] for r in records], np.float64)
        if (affine.shape != (7, 3, 3) or not np.isfinite(affine).all()
                or not np.array_equal(affine[:, 2], np.tile([0., 0., 1.], (7, 1)))
                or (np.linalg.det(affine[:, :2, :2]) <= 0).any()):
            raise ValueError('Finite orientation-preserving native affine maps required')
        for t in range(6, -1, -1):
            past[t] = past[t+1]@affine[t]
        for t in range(12):
            prediction[t] = np.linalg.matrix_power(affine[-1], t+1)
    if not np.isfinite(past).all() or not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite causal camera composition')
    return dict(past=past, prediction=prediction, supported_pairs=int(supported.sum()),
        active=bool(supported.all()))


def relative_maps(maps, origins):
    m, o = np.asarray(maps, np.float64), np.asarray(origins, np.float64)
    if (m.ndim != 4 or m.shape[2:] != (3, 3) or o.shape != (len(m), 2)
            or not np.isfinite(m).all() or not np.isfinite(o).all()
            or not np.array_equal(m[:, :, 2], np.broadcast_to([0., 0., 1.], m[:, :, 2].shape))):
        raise ValueError('Aligned native affine maps and observed origins required')
    local = m[:, :, :2].copy()
    local[:, :, :, 2] = np.einsum('ntij,nj->nti', m[:, :, :2, :2], o)+m[:, :, :2, 2]-o[:, None]
    local = local.astype(np.float32)
    if not np.isfinite(local).all():
        raise FloatingPointError('Nonfinite image-coordinate camera maps')
    return local


def pack_inputs(geometry, origins, observed_maps, predicted_maps, arm):
    if arm not in ARMS:
        raise ValueError('Declared camera arm required')
    n = len(geometry)
    if np.shape(observed_maps) != (n, 8, 3, 3) or np.shape(predicted_maps) != (n, 12, 3, 3):
        raise ValueError('Eight observed and twelve past-extrapolated camera maps required')
    if arm == 'identity':
        past = np.broadcast_to(np.eye(3)[:2], (n, 8, 2, 3)).astype(np.float32).copy()
        future = np.broadcast_to(np.eye(3)[:2], (n, 12, 2, 3)).astype(np.float32).copy()
    else:
        past, future = relative_maps(observed_maps, origins), relative_maps(predicted_maps, origins)
    return dict(pack_geometry(geometry), observed_camera_maps=torch.from_numpy(past),
        predicted_camera_maps=torch.from_numpy(future))


def camera_cv(history, observed_maps, predicted_maps):
    h, past, future = map(lambda x: np.asarray(x, np.float64), (history, observed_maps, predicted_maps))
    if h.shape != (len(h), 8, 2) or past.shape != (len(h), 8, 3, 3) or future.shape != (len(h), 12, 3, 3):
        raise ValueError('Complete aligned causal history and camera operators required')
    stable = np.einsum('ntij,ntj->nti', past[:, :, :2, :2], h)+past[:, :, :2, 2]
    velocity = stable[:, -1]-stable[:, -2]
    path = stable[:, -1, None]+np.arange(1, 13)[None, :, None]*velocity[:, None]
    result = np.einsum('ntij,ntj->nti', future[:, :, :2, :2], path)+future[:, :, :2, 2]
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite camera-compensated CV')
    return result


class CameraFrameForecaster(SourceForecaster):
    def __init__(self, architecture, baseline_index, arm):
        if arm not in ARMS or architecture.get('family') != 'eqmotion_fixed_head':
            raise ValueError('Frozen EqMotion architecture and declared camera arm required')
        super().__init__(architecture, baseline_index)
        self.arm = arm

    def forward(self, inputs):
        if set(inputs) != INPUT_KEYS|CAMERA_KEYS:
            raise ValueError('Only causal geometry and past-derived camera operators accepted')
        safe = {k: inputs[k] for k in INPUT_KEYS}
        h, neighbors = safe['history'], safe['neighbors']
        past, future = inputs['observed_camera_maps'], inputs['predicted_camera_maps']
        if (past.shape != (len(h), 8, 2, 3) or future.shape != (len(h), 12, 2, 3)
                or not torch.isfinite(past).all() or not torch.isfinite(future).all()
                or not torch.equal(past[:, -1], torch.eye(3, device=past.device, dtype=past.dtype)[:2].expand(len(h), -1, -1))):
            raise ValueError('Finite maps with an identity current frame required')
        if self.arm == 'identity':
            if (not torch.equal(past, torch.eye(3, device=past.device, dtype=past.dtype)[:2].expand_as(past))
                    or not torch.equal(future, torch.eye(3, device=future.device, dtype=future.dtype)[:2].expand_as(future))):
                raise ValueError('The identity control cannot receive a nonidentity operator')
            return super().forward(safe)
        aligned = torch.isclose(neighbors[..., 2], h[:, None, :, 2], atol=1e-6, rtol=1e-6)
        if not aligned[safe['neighbor_mask']].all():
            raise ValueError('Observed neighbor times must match the native history grid')
        xy = torch.einsum('ntij,ntj->nti', past[..., :2], h[..., :2])+past[..., 2]
        nxy = torch.einsum('ntij,nktj->nkti', past[..., :2], neighbors[..., :2])+past[:, None, :, :, 2]
        nxy = torch.where(safe['neighbor_mask'][..., None], nxy, neighbors[..., :2])
        safe['history'] = torch.cat((xy, h[..., 2:]), -1)
        safe['neighbors'] = torch.cat((nxy, neighbors[..., 2:]), -1)
        canonical_prediction = super().forward(safe)
        return torch.einsum('ntij,ntj->nti', future[..., :2], canonical_prediction)+future[..., 2]
