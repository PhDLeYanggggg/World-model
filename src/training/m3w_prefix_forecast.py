"""Matched forecast fitting with a fixed, available-label prefix objective."""
from __future__ import annotations

import math
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import numpy as np
import torch
from src.world_model.m3w_native_forecast import pack_geometry, draw_batch, masked_objective

ARMS = ('available12', 'prefix_balanced')
PREFIXES = (1, 3, 6)


def objective(prediction, target, valid, factors, arm):
    loss, full = masked_objective(prediction, target, valid, factors)
    if arm == 'available12':
        return loss, full
    if arm != 'prefix_balanced':
        raise ValueError('Only the two fixed matched losses are registered')
    safe = torch.where(valid[..., None], target, 0.)
    distance = torch.where(valid, torch.linalg.vector_norm(prediction-safe, dim=-1), 0.)
    counts = torch.stack([valid[:, :k].sum(1) for k in PREFIXES], 1)
    sums = torch.stack([distance[:, :k].sum(1) for k in PREFIXES], 1)
    known = counts > 0
    prefix = (sums/counts.clamp_min(1)).sum(1)/known.sum(1).clamp_min(1)
    # A row with labels only after step6 keeps full weight on its available loss.
    early = torch.where(known.any(1), prefix, full)
    return ((.5*full+.5*early)*factors).mean(), full


def fit(model, data, fold, *, arm, seed, settings, identity, directory,
        resume=False, stop_at=None, heartbeat):
    """Original sampling/optimizer/checkpoint layout, with one explicit loss change."""
    if arm not in ARMS:
        raise ValueError('Fixed loss arm required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.pt'
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    rng = torch.Generator().manual_seed(seed+7919)
    step, seconds, losses = 0, 0., []
    draws = np.zeros(len(data['sites']), np.int64)
    if checkpoint.exists():
        if not resume:
            raise FileExistsError('Existing checkpoint requires explicit --resume')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['settings'] != settings or saved['seed'] != seed:
            raise ValueError('Checkpoint identity/settings changed')
        np.testing.assert_array_equal(saved['train_ids'], fold['train_ids'])
        np.testing.assert_array_equal(saved['factors'], fold['factors'])
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step, seconds, losses, draws = saved['step'], saved['seconds'], saved['losses'], saved['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or min(limit, settings['batch_size'], settings['checkpoint_every'], settings['heartbeat_every']) <= 0:
        raise ValueError('Invalid registered training/resume budget')
    model.train()
    allowed = np.zeros(len(data['sites']), bool); allowed[fold['train_ids']] = True

    def save():
        state = dict(identity=identity, settings=settings, seed=seed, step=step,
            seconds=seconds+time.monotonic()-started, losses=losses, draws=draws,
            train_ids=fold['train_ids'], factors=fold['factors'], model=model.state_dict(),
            optimizer=opt.state_dict(), sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state())
        temp = checkpoint.with_suffix('.tmp')
        torch.save(state, temp); os.replace(temp, checkpoint)

    try:
        while step < limit:
            ids = draw_batch(fold['groups'], settings['batch_size'], rng)
            if not allowed[ids].all():
                raise ValueError('Held source entered training batch')
            progress = step/max(settings['steps']-1, 1)
            lr = settings['learning_rate']*(settings['minimum_lr_ratio']+
                (1-settings['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
            for group in opt.param_groups:
                group['lr'] = lr
            inputs = pack_geometry(data['geometry'][ids])
            target = torch.from_numpy(data['target'][ids].copy())
            mask = torch.from_numpy(data['valid'][ids].copy())
            factors = torch.tensor(fold['factors'][ids], dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            loss, raw_ade = objective(model(inputs), target, mask, factors, arm)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), learning_rate=lr,
                    supported_batch_rows=int(mask.any(1).sum()), mean_past_normalized_ADE=float(raw_ade.detach().mean()))
                losses.append(row)
                heartbeat(state='training', **row, fit_seconds=seconds+time.monotonic()-started)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_update=step, checkpoint=str(checkpoint))
        raise
    model.eval()
    return dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, losses=losses, total_draws=int(draws.sum()),
        unique_training_rows=int((draws > 0).sum()), held_rows_sampled=int(draws[fold['held_ids']].sum()),
        parameters=sum(p.numel() for p in model.parameters()))
