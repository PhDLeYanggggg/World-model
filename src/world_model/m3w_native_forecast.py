"""Matched source-only forecast fitting; input conditioning is not error weighting."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')

import numpy as np
import torch


def pack_geometry(geometry):
    g = np.asarray(geometry, np.float32)
    if g.ndim != 2 or g.shape[1] != 476 or not np.isfinite(g).all():
        raise ValueError('Finite frozen 476-column past geometry required')
    n = len(g)
    htime = g[:, 16:24]
    ntime = g[:, 166:230].reshape(n, 8, 8)
    mask = g[:, 230:294].reshape(n, 8, 8)
    if (not np.isin(mask, [0, 1]).all() or np.any(htime > 0)
            or np.any(np.diff(htime, axis=1) <= 0) or np.any(htime[:, -1] != 0)
            or np.any(ntime[mask.astype(bool)] > 0)):
        raise ValueError('Only query-aligned past timestamps and binary masks allowed')
    return dict(
        history=torch.from_numpy(np.concatenate((g[:, :16].reshape(n, 8, 2), htime[..., None]), -1)),
        history_mask=torch.ones(n, 8, dtype=torch.bool),
        neighbors=torch.from_numpy(np.concatenate((g[:, 38:166].reshape(n, 8, 8, 2), ntime[..., None]), -1)),
        neighbor_mask=torch.from_numpy(mask.astype(bool)),
        baseline=torch.from_numpy(g[:, 332:356].reshape(n, 12, 2).copy()),
        prediction_time=torch.arange(1, 13, dtype=torch.float32)[None].expand(n, -1)/12,
        request_mask=torch.ones(n, 12, dtype=torch.bool))


def fold_design(data, held_site, objective):
    if objective not in ('past_normalized', 'native_coordinate'):
        raise ValueError('Only the registered loss arms are supported')
    sites, valid = np.asarray(data['sites']), np.asarray(data['valid'])
    if valid.shape != (len(sites), 12) or valid.dtype != bool or held_site not in sites:
        raise ValueError('Aligned labels and explicit held scene required')
    train, held = np.flatnonzero(sites != held_site), np.flatnonzero(sites == held_site)
    if not len(train) or not len(held):
        raise ValueError('Nonempty disjoint training/held sites required')
    factors = np.zeros(len(sites), np.float64)
    groups, normalizers, native_cv = [], {}, []
    for site in sorted(set(sites[train])):
        ids = np.flatnonzero(sites == site)
        mask, target, scale = valid[ids], data['target'][ids], np.asarray(data['scale'][ids], float)
        if not np.isfinite(target[mask]).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Finite training labels and positive past scales required')
        count = mask.sum(1); supported = count > 0
        if not supported.any():
            raise ValueError('No supported training labels in a scene')
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2).astype(float)
        distance = np.linalg.norm(baseline-np.where(mask[..., None], target, 0).astype(float), axis=-1)
        error = np.divide(np.where(mask, distance, 0).sum(1), count,
                          out=np.zeros(len(ids)), where=supported)
        units = scale if objective == 'native_coordinate' else np.ones(len(ids))
        mean_cv = float((error[supported]*units[supported]).mean())
        if mean_cv <= 0 or not np.isfinite(mean_cv):
            raise ValueError('Positive training CV loss mean required')
        correction = len(ids)/int(supported.sum())
        factors[ids] = correction*units/mean_cv
        groups.append(ids)
        native_cv.extend((error[supported]*scale[supported]).tolist())
        normalizers[site] = dict(rows=len(ids), supported=int(supported.sum()),
            loss_CV_mean=mean_cv, indexed_to_supported_correction=correction)
    return dict(train_ids=train, held_ids=held, groups=groups, factors=factors,
                normalizers=normalizers, hard_cut=float(np.quantile(native_cv, .75)), objective=objective)


def masked_objective(prediction, target, valid, factors):
    if (prediction.shape != target.shape or valid.shape != target.shape[:2]
            or valid.dtype != torch.bool or factors.shape != (len(target),)
            or not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all()
            or not torch.isfinite(factors).all() or (factors < 0).any()):
        raise ValueError('Finite predictions, supported labels and registered loss factors required')
    safe = torch.where(valid[..., None], target, 0.)
    distance = torch.linalg.vector_norm(prediction-safe, dim=-1)
    per_row = torch.where(valid, distance, 0.).sum(1)/valid.sum(1).clamp_min(1)
    return (per_row*factors).mean(), per_row


def draw_batch(groups, size, generator):
    choices = torch.randint(len(groups), (size,), generator=generator).numpy()
    batch = np.empty(size, np.int64)
    for k, ids in enumerate(groups):
        where = np.flatnonzero(choices == k)
        offsets = torch.randint(len(ids), (len(where),), generator=generator).numpy()
        batch[where] = ids[offsets]
    return batch


def fit_trial(model, data, fold, *, seed, settings, identity, directory,
              resume=False, stop_at=None, heartbeat):
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
    allowed = np.zeros(len(data['sites']), bool)
    allowed[fold['train_ids']] = True

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
                raise ValueError('Held source entered the training batch')
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
            loss, raw_ade = masked_objective(model(inputs), target, mask, factors)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
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
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_update=step,
                  checkpoint=str(checkpoint))
        raise
    model.eval()
    return dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, losses=losses,
        total_draws=int(draws.sum()), unique_training_rows=int((draws > 0).sum()),
        held_rows_sampled=int(draws[fold['held_ids']].sum()),
        parameters=sum(p.numel() for p in model.parameters()))


def predict(model, data, ids, batch_size=128):
    model.eval(); result = []
    with torch.no_grad():
        for offset in range(0, len(ids), batch_size):
            result.append(model(pack_geometry(data['geometry'][ids[offset:offset+batch_size]])).numpy())
    return np.concatenate(result)
