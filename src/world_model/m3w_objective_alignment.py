"""Matched fit-only objective controls over the frozen geometry forecaster."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch import nn

ARMS = {
    'row_log': ('row', 'log'),
    'row_ade': ('row', 'ade'),
    'scene_log': ('scene', 'log'),
    'scene_ade': ('scene', 'ade'),
    'scene_ade_harm': ('scene', 'ade_harm'),
}


def supported_standardization(train, queries):
    """Learn all moments/support on training inputs, never query statistics."""
    train, queries = np.asarray(train), np.asarray(queries)
    if train.ndim != 2 or queries.ndim != 2 or train.shape[1] != queries.shape[1] or not len(train):
        raise ValueError('Compatible nonempty training features required')
    if not np.isfinite(train).all() or not np.isfinite(queries).all():
        raise ValueError('Nonfinite geometry')
    mean, std = train.mean(0), np.maximum(train.std(0), 1e-6)
    constant = np.ptp(train, axis=0) == 0
    normalized = np.clip((queries - mean) / std, -10, 10)
    normalized[:, constant] = 0.
    return normalized.astype(np.float32), {'mean': mean, 'std': std, 'constant': constant}


def sampling_weights(folds, mode):
    folds = np.asarray(folds)
    if folds.ndim != 1 or not len(folds) or mode not in ('row', 'scene'):
        raise ValueError('Nonempty scene memberships and registered sampling required')
    weights = np.ones(len(folds), np.float64)
    if mode == 'scene':
        for fold in np.unique(folds):
            mask = folds == fold
            weights[mask] = 1. / mask.sum()
    return torch.as_tensor(weights / weights.sum(), dtype=torch.float64)


def geometry_prediction(model, geometry, observed, baseline):
    """Exactly the old geometry arm, without repeatedly loading unused RGB."""
    if observed.shape != (len(geometry), 8) or baseline.shape != (len(geometry), 12, 2):
        raise ValueError('Registered eight-observation/twelve-target shapes required')
    latent = geometry.new_zeros((len(geometry), 128))
    return baseline + model.output(model.fusion(torch.cat((geometry, observed, latent), 1))).reshape(-1, 12, 2)


def objective_loss(prediction, target, baseline, objective):
    if prediction.shape != target.shape or baseline.shape != target.shape or target.shape[1:] != (12, 2):
        raise ValueError('Complete 12-step labels required')
    error = torch.linalg.vector_norm(prediction - target, dim=-1).mean(-1)
    reference = torch.linalg.vector_norm(baseline.detach() - target, dim=-1).mean(-1)
    excess = (error - reference).clamp_min(0)
    if objective == 'log':
        loss = error.log1p().mean()
    elif objective == 'ade':
        loss = error.mean()
    elif objective == 'ade_harm':
        loss = (error + excess).mean()
    else:
        raise ValueError('Unregistered objective')
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective')
    return loss, {'ADE': float(error.detach().mean()), 'CV_ADE': float(reference.detach().mean()),
                  'positive_harm': float(excess.detach().mean())}


def fit_objective(model, batch, train_ids, train_folds, *, arm, config, seed,
                  identity, checkpoint, heartbeat, stop_at=None):
    sampling, objective = ARMS[arm]
    weights = sampling_weights(train_folds, sampling)
    if len(train_ids) != len(weights):
        raise ValueError('Training membership mismatch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed + 7919)
    step, losses, elapsed = 0, [], 0.
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity or state['config'] != config or state['arm'] != arm:
            raise ValueError('Resume identity/config/arm mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        step, losses, elapsed = state['step'], state['losses'], state['fit_seconds']
    initial_step = step
    limit = min(stop_at, config['updates']) if stop_at is not None else config['updates']
    started = time.monotonic()
    model.train()
    while step < limit:
        chosen = torch.multinomial(weights, config['batch_size'], replacement=True, generator=generator)
        x, observed, baseline, target = batch(train_ids[chosen])
        optimizer.zero_grad(set_to_none=True)
        loss, detail = objective_loss(geometry_prediction(model, x, observed, baseline), target, baseline, objective)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step == 1 or step % 100 == 0:
            losses.append({'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(grad), **detail})
        if step % config['checkpoint_every'] == 0 or step == limit:
            seconds = elapsed + time.monotonic() - started
            state = dict(identity=identity, config=config, arm=arm, model=model.state_dict(),
                optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(), step=step, losses=losses, fit_seconds=seconds)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(state, temporary)
            os.replace(temporary, checkpoint)
            heartbeat({'state': 'training' if step < config['updates'] else 'fit_complete',
                       'pid': os.getpid(), 'step': step, 'fit_seconds': seconds, 'loss': float(loss.detach())})
    model.eval()
    return {'step': step, 'new_updates_this_invocation': step - initial_step, 'losses': losses,
            'fit_seconds': elapsed + time.monotonic() - started, 'complete': step == config['updates']}
