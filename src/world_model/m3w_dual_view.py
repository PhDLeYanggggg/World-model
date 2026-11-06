"""Native-coordinate forecasting with additional per-node observed-view features."""
import platform
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import torch
from torch import nn
from src.world_model.m3w_camera_frame import relative_maps
from src.world_model.m3w_context_conditioning import INPUT_KEYS, condition_inputs
from src.world_model.m3w_european_source_forecast import SourceForecaster, baseline_torch
from src.world_model.m3w_native_forecast import pack_geometry

ARMS = ('raw_control', 'native_node', 'camera_node')


def pack_inputs(geometry, origins, observed_maps):
    return dict(pack_geometry(geometry), observed_camera_maps=torch.from_numpy(relative_maps(observed_maps, origins)))


def aligned_node_features(context, scale, maps):
    x, velocity = context['x'], context['velocity']
    aligned = torch.einsum('ntij,nktj->nkti', maps[..., :2], x)+maps[:, None, :, :, 2]/scale[:, None, None, None]
    aligned_velocity = torch.zeros_like(aligned)
    aligned_velocity[:, :, 1:] = aligned[:, :, 1:]-aligned[:, :, :-1]
    aligned_velocity[:, :, 0] = aligned_velocity[:, :, 1]
    # All three streams use observed points; the native geometry is never replaced.
    features = torch.cat((velocity.flatten(-2), aligned_velocity.flatten(-2), (aligned-x).flatten(-2)), -1)
    if not torch.isfinite(features).all(): raise ValueError('Nonfinite observed node-view features')
    return torch.asinh(features)


class DualViewForecaster(SourceForecaster):
    def __init__(self, architecture, baseline_index, arm):
        if (arm not in ARMS or architecture.get('family') != 'eqmotion_fixed_head'
                or architecture.get('input_conditioning') != 'observed_joint_max_norm'):
            raise ValueError('Declared dual-view arm and pinned conditioned architecture required')
        super().__init__(architecture, baseline_index); self.arm = arm
        rng = torch.get_rng_state()
        self.node_encoder = nn.Sequential(nn.Linear(48, 32, bias=False), nn.SiLU(), nn.Linear(32, 8, bias=False))
        nn.init.zeros_(self.node_encoder[-1].weight)
        torch.set_rng_state(rng)
        if arm == 'raw_control': self.node_encoder.requires_grad_(False)

    def forward(self, inputs):
        if set(inputs) != INPUT_KEYS|{'observed_camera_maps'}:
            raise ValueError('Only native past inputs and observed camera maps accepted')
        maps = inputs['observed_camera_maps']; n = len(inputs['history'])
        identity = torch.eye(3, dtype=maps.dtype, device=maps.device)[:2]
        if (maps.shape != (n, 8, 2, 3) or not torch.isfinite(maps).all()
                or not torch.equal(maps[:, -1], identity.expand(n, -1, -1))):
            raise ValueError('Finite observed maps with identity current frame required')
        safe = {k: inputs[k] for k in INPUT_KEYS}
        if self.arm == 'raw_control': return super().forward(safe)
        safe['baseline'] = baseline_torch(safe['history'], self.baseline_index)
        conditioned, scale = condition_inputs(safe); adapter = self.predictor.predictor
        context = adapter.prepare_context(conditioned)
        selected_maps = identity.expand_as(maps) if self.arm == 'native_node' else maps
        features = aligned_node_features(context, scale, selected_maps)
        valid_nodes = torch.arange(context['x'].shape[1], device=maps.device)[None] < context['num_valid'][:, None]
        offset = .1*torch.tanh(self.node_encoder(features))*valid_nodes[..., None]
        forecasts, _ = adapter.core(context['speed']+offset, context['x'], context['velocity'], context['num_valid'])
        return forecasts[:, 0, 0 if adapter.prune_unused_heads else adapter.fixed_head]*scale[:, None, None]
