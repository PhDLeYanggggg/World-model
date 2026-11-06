"""Past-image conditioning of the pinned K1 trajectory core, not a residual head."""
import platform
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import numpy as np
import torch
from torch import nn
from src.world_model.m3w_european_source_forecast import SourceForecaster, baseline_torch
from src.world_model.m3w_context_conditioning import INPUT_KEYS, condition_inputs
from src.world_model.m3w_native_forecast import pack_geometry

ARMS = ('raw_control', 'quality_geometry', 'motion_geometry')
CONTEXT_COLUMNS = 46


def normalizer(values, fold):
    x = np.asarray(values); ids = np.asarray(fold['train_ids']); groups = fold['groups']
    if (x.shape != (len(x), CONTEXT_COLUMNS) or not np.isfinite(x).all()
            or not len(ids) or len(np.unique(ids)) != len(ids) or not groups
            or not np.array_equal(np.sort(np.concatenate(groups)), np.sort(ids))
            or np.intersect1d(ids, fold['held_ids']).size):
        raise ValueError('Complete fitting-only past contexts and source groups required')
    mean = np.mean([x[g].mean(0, dtype=np.float64) for g in groups], axis=0)
    variance = np.mean([np.mean((x[g]-mean)**2, axis=0) for g in groups], axis=0)
    return dict(mean=mean, std=np.sqrt(variance).clip(1e-6))


def pack_inputs(geometry, context):
    c = np.asarray(context, np.float32)
    if c.shape != (len(geometry), CONTEXT_COLUMNS) or not np.isfinite(c).all():
        raise ValueError('Finite declared past-only image context required')
    return dict(pack_geometry(geometry), image_context=torch.from_numpy(c.copy()))


class VisualSourceForecaster(SourceForecaster):
    def __init__(self, architecture, baseline_index, mean, std, arm):
        if arm not in ARMS or architecture.get('family') != 'eqmotion_fixed_head' or architecture.get('input_conditioning') != 'observed_joint_max_norm':
            raise ValueError('Declared image arm and unchanged conditioned EqMotion required')
        mean, std = np.asarray(mean), np.asarray(std)
        if mean.shape != (46,) or std.shape != (46,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Fitting-only finite46-column normalizer required')
        super().__init__(architecture, baseline_index)
        self.arm = arm
        self.register_buffer('image_mean', torch.as_tensor(mean.astype(np.float32)))
        self.register_buffer('image_std', torch.as_tensor(std.astype(np.float32)))
        # Preserve the exact original core RNG stream for the masked control.
        rng = torch.get_rng_state()
        self.image_encoder = nn.Sequential(nn.Linear(46, 32, bias=False), nn.SiLU(), nn.Linear(32, 8, bias=False))
        nn.init.zeros_(self.image_encoder[-1].weight)
        torch.set_rng_state(rng)
        if arm == 'raw_control': self.image_encoder.requires_grad_(False)

    def forward(self, inputs):
        if set(inputs) != INPUT_KEYS|{'image_context'}:
            raise ValueError('Only original past inputs plus declared image context accepted')
        c = inputs['image_context']
        if c.shape != (len(inputs['history']), 46) or not torch.isfinite(c).all():
            raise ValueError('Finite causal image context required')
        safe = {k: inputs[k] for k in INPUT_KEYS}
        safe['baseline'] = baseline_torch(safe['history'], self.baseline_index)
        conditioned, scale = condition_inputs(safe)
        adapter = self.predictor.predictor
        context = adapter.prepare_context(conditioned)
        z = torch.zeros_like(c) if self.arm == 'raw_control' else ((c-self.image_mean)/self.image_std).clamp(-8, 8)
        offset = .1*torch.tanh(self.image_encoder(z))
        nodes = torch.arange(context['x'].shape[1], device=c.device)[None] < context['num_valid'][:, None]
        features = context['speed']+offset[:, None]*nodes[..., None]
        forecasts, _ = adapter.core(features, context['x'], context['velocity'], context['num_valid'])
        return forecasts[:, 0, 0 if adapter.prune_unused_heads else adapter.fixed_head]*scale[:, None, None]
