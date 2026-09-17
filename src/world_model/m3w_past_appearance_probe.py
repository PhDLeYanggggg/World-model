"""Small fit-only appearance ablation with explicit input/label separation."""
from __future__ import annotations

import os
from pathlib import Path
import time

import torch
from torch import nn


ARMS = ('geometry', 'current_rgb', 'past_rgb')


class PastAppearanceProbe(nn.Module):
    def __init__(self, geometry_dim=32):
        super().__init__()
        self.geometry_dim = geometry_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(64, 16), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(geometry_dim+8+8*16, 64), nn.SiLU(),
                                    nn.Linear(64, 32), nn.SiLU())
        self.trajectory = nn.Linear(32, 24)
        self.start = nn.Linear(32, 1)
        nn.init.zeros_(self.trajectory.weight)
        nn.init.zeros_(self.trajectory.bias)

    def forward(self, geometry, images, mask, arm):
        n = len(geometry)
        if (arm not in ARMS or geometry.shape != (n, self.geometry_dim)
                or images.shape != (n, 8, 3, 32, 32) or mask.shape != (n, 8)):
            raise ValueError('Frozen past-input schema or arm mismatch')
        if arm == 'geometry':
            visual = geometry.new_zeros((n, 8, 16))
        elif arm == 'current_rgb':
            last = self.image_encoder(images[:, -1]) * mask[:, -1, None]
            visual = last[:, None].expand(-1, 8, -1)
        else:
            visual = self.image_encoder(images.reshape(n*8, 3, 32, 32)).reshape(n, 8, 16)
            visual = visual * mask[..., None]
        z = self.fusion(torch.cat([geometry, mask.to(geometry.dtype), visual.reshape(n, -1)], dim=1))
        return self.trajectory(z).reshape(n, 12, 2), self.start(z).squeeze(-1)


def pixel_delta_to_native(delta, image_xy, homography, pixel_scale=96.0):
    """Supplied row/column H, not verified physical calibration. Zero maps to zero."""
    delta, xy, h = delta.double()*pixel_scale, image_xy.double(), homography.double()
    if delta.ndim != 3 or delta.shape[1:] != (12, 2) or xy.shape != (len(delta), 2) or h.shape != (len(delta), 3, 3):
        raise ValueError('Projection input shape mismatch')
    future = xy[:, None] + delta
    p = torch.cat([future[..., [1, 0]], torch.ones_like(future[..., :1])], dim=-1)
    p0 = torch.cat([xy[:, [1, 0]], torch.ones_like(xy[:, :1])], dim=-1)
    q = torch.einsum('nij,ntj->nti', h, p)
    q0 = torch.einsum('nij,nj->ni', h, p0)
    if torch.any(q[..., 2].abs() < 1e-8) or torch.any(q0[:, 2].abs() < 1e-8):
        raise FloatingPointError('Prediction crosses supplied projection singularity')
    return q[..., :2]/q[..., 2:] - (q0[:, :2]/q0[:, 2:])[:, None]


def fit_probe(model, data, train, config, identity, checkpoint, *, stop_at=None, heartbeat=None):
    """Uniform row minibatches; exact optimizer/RNG resume; no held labels here."""
    checkpoint = Path(checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(identity['seed']+7919)
    losses, step, elapsed = [], 0, 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity or state['config'] != config:
            raise ValueError('Resume identity/config changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        losses, step, elapsed = state['losses'], state['step'], state['fit_seconds']
    start_step, started = step, time.monotonic()
    model.train()
    train = torch.as_tensor(train, dtype=torch.long)

    def save():
        temporary = checkpoint.with_suffix('.tmp.pt')
        torch.save({'identity': identity, 'config': config, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'sampler_rng': generator.get_state(),
                    'torch_rng': torch.get_rng_state(), 'step': step, 'losses': losses,
                    'fit_seconds': elapsed+time.monotonic()-started}, temporary)
        os.replace(temporary, checkpoint)

    limit = config['updates'] if stop_at is None else min(stop_at, config['updates'])
    for step0 in range(step, limit):
        ids = train[torch.randint(len(train), (config['batch_size'],), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        delta, logit = model(data['geometry'][ids], data['images'][ids], data['mask'][ids], identity['arm'])
        prediction = pixel_delta_to_native(delta, data['image_xy'][ids], data['homography'][ids])
        residual = prediction-data['target_native'][ids]
        distance = torch.sqrt(residual.square().sum(-1)+config['distance_epsilon']**2)-config['distance_epsilon']
        trajectory_loss = (distance.mean(1)/data['parent_scale'][ids]).mean()*config['loss_numeric_scale']
        start_loss = nn.functional.binary_cross_entropy_with_logits(logit, data['start'][ids])
        loss = trajectory_loss+config['start_loss_weight']*start_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite appearance loss')
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        step = step0+1
        losses.append({'step':step, 'total':float(loss.detach()), 'trajectory':float(trajectory_loss.detach()),
                       'start_bce':float(start_loss.detach()), 'gradient_norm':float(grad)})
        if step % config['checkpoint_every'] == 0 or step == limit:
            save()
            if heartbeat is not None:
                heartbeat(step, losses[-1], elapsed+time.monotonic()-started)
    return {'step': step, 'new_updates': step-start_step, 'losses': losses,
            'fit_seconds': elapsed+time.monotonic()-started, 'complete': step == config['updates']}
