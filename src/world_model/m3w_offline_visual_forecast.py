"""Fit-only offline-annotation visual forecasting, without changing old protocols."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the arm64 .venv-pytorch environment')

import numpy as np
import torch
from torch import nn

ARMS = ('geometry', 'mask_only', 'current_rgb', 'past_rgb')


def geometry_features(inputs):
    """Explicit allowlist: no labels, source control provenance or identities."""
    horizon = float(inputs['prediction_frame_offsets'][-1])
    mask = inputs['neighbor_mask']
    neighbors = np.where(mask[..., None], inputs['neighbor_xy'], 0.)
    times = np.where(mask, inputs['neighbor_frame_offsets'] / horizon, 0.)
    return np.concatenate((inputs['history_xy'].ravel(),
        inputs['history_frame_offsets'] / horizon, inputs['history_velocity'].ravel(),
        neighbors.ravel(), times.ravel(), mask.ravel(), inputs['causal_features'],
        inputs['baseline_rollouts'].ravel())).astype(np.float32)


def normalized_targets(history, future, transform):
    if history.shape != (8, 4) or future.shape != (12, 2):
        raise ValueError('The registered complete 8-to-12 task is required')
    return ((future - transform['origin_xy']) @ transform['rotation'] /
            transform['scale']).astype(np.float32)


class OfflineVisualForecast(nn.Module):
    """Matched small CNN/MLP; zero-initialized CV skip in past-normalized coordinates."""
    def __init__(self, geometry_dim):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(4, 8, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d(2),
            nn.Flatten(), nn.Linear(64, 16), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(geometry_dim + 8 + 128, 128), nn.SiLU(),
                                    nn.Linear(128, 64), nn.SiLU())
        self.output = nn.Linear(64, 24)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, geometry, rgb, coverage, baseline, arm):
        if arm not in ARMS:
            raise ValueError('Unknown registered arm')
        if rgb.shape[1:] != (8, 3, 32, 32) or coverage.shape[1:] != (8, 1, 32, 32):
            raise ValueError('Expected eight masked 32x32 images')
        rgb = torch.where(coverage > 0, rgb, torch.zeros_like(rgb))
        observed = coverage.mean((2, 3, 4))
        if arm == 'geometry':
            latent = geometry.new_zeros((len(geometry), 8, 16))
        else:
            if arm == 'mask_only':
                rgb = torch.zeros_like(rgb)
            if arm == 'current_rgb':
                rgb, cov = rgb[:, -1:], coverage[:, -1:]
            else:
                cov = coverage
            images = torch.cat((rgb, cov), 2)
            latent = self.image(images.flatten(0, 1)).reshape(len(geometry), -1, 16)
            valid = cov.sum((2, 3, 4)) > 0
            latent = latent * valid[..., None]
            if arm == 'current_rgb':
                latent = latent.expand(-1, 8, -1)
        context = torch.cat((geometry, observed, latent.flatten(1)), 1)
        return baseline + self.output(self.fusion(context)).reshape(-1, 12, 2)


def fit_model(model, batch, train_ids, *, arm, config, seed, identity, checkpoint,
              heartbeat, stop_at=None):
    """Exact resume includes optimizer and minibatch RNG, not just model weights."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed + 7919)
    step, losses, prior_seconds = 0, [], 0.
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity or state['config'] != config:
            raise ValueError('Resume identity/config changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        step, losses, prior_seconds = state['step'], state['losses'], state['fit_seconds']
    limit = min(config['updates'], stop_at) if stop_at is not None else config['updates']
    started = time.monotonic()
    model.train()
    while step < limit:
        ids = train_ids[torch.randint(len(train_ids), (config['batch_size'],), generator=generator)]
        values = batch(ids)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*values[:4], arm)
        distance = torch.linalg.vector_norm(prediction - values[4], dim=-1).mean(-1)
        loss = torch.log1p(distance).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite forecasting loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step == 1 or step % 100 == 0:
            losses.append({'step': step, 'log1p_primary_ADE_loss': float(loss.detach())})
        if step % config['checkpoint_every'] == 0 or step == limit:
            elapsed = prior_seconds + time.monotonic() - started
            state = dict(identity=identity, config=config, model=model.state_dict(),
                optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(), step=step, losses=losses, fit_seconds=elapsed)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(state, temporary)
            os.replace(temporary, checkpoint)
            heartbeat({'pid': os.getpid(), 'state': 'training' if step < config['updates'] else 'fit_complete',
                       'step': step, 'fit_seconds': elapsed, 'loss': float(loss.detach())})
    model.eval()
    return {'step': step, 'losses': losses, 'fit_seconds': prior_seconds + time.monotonic() - started,
            'complete': step == config['updates']}


def forecast_metrics(prediction, target, baseline, scale, easy_threshold):
    error = np.linalg.norm(prediction.astype(np.float64) - target, axis=-1)
    ref = np.linalg.norm(baseline.astype(np.float64) - target, axis=-1)
    ade, reference = error.mean(1), ref.mean(1)
    easy = reference <= easy_threshold
    def relative(a, b):
        return float(100 * (1 - a.mean() / b.mean())) if len(a) and b.mean() > 1e-12 else None
    return {'rows': len(ade), 'primary_ADE': float(ade.mean()), 'reference_ADE': float(reference.mean()),
        'improvement_percent': relative(ade, reference), 'FDE': float(error[:, -1].mean()),
        'native_ADE_diagnostic': float((ade * scale).mean()),
        'native_FDE_diagnostic': float((error[:, -1] * scale).mean()),
        'easy_rows': int(easy.sum()), 'easy_degradation_percent': (
            -relative(ade[easy], reference[easy]) if relative(ade[easy], reference[easy]) is not None else None),
        'easy_absolute_harm': float((ade[easy] - reference[easy]).mean()) if easy.any() else None,
        'mean_harm_over_reference': float((ade - reference).mean()),
        'tail_ADE_p95': float(np.quantile(ade, .95)), 'forecast_nonfinite': int((~np.isfinite(error)).sum())}
