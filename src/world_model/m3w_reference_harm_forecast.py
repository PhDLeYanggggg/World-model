"""Source-only forecast loss repair; reference errors are supervision, not inputs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')

import numpy as np
import torch
from src.world_model.m3w_european_source_forecast import baseline_torch
from src.world_model.m3w_native_forecast import draw_batch, masked_objective, pack_geometry


def reference_harm_objective(prediction, target, valid, factors, selected, cv,
                             *, selected_weight, cv_weight):
    weights = (selected_weight, cv_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Finite nonnegative harm weights required')
    native, ade = masked_objective(prediction, target, valid, factors)
    if not any(weights):
        return native, ade, {}
    if selected.shape != prediction.shape or cv.shape != prediction.shape:
        raise ValueError('Reference trajectories must have the same support shape')
    with torch.no_grad():
        _, selected_ade = masked_objective(selected.detach(), target, valid, factors)
        _, cv_ade = masked_objective(cv.detach(), target, valid, factors)
    selected_harm = (torch.relu(ade-selected_ade)*factors).mean()
    cv_harm = (torch.relu(ade-cv_ade)*factors).mean()
    loss = native + selected_weight*selected_harm + cv_weight*cv_harm
    return loss, ade, dict(native_loss=native, selected_positive_harm=selected_harm,
                          cv_positive_harm=cv_harm)


def fit_trial(model, data, fold, *, seed, settings, identity, directory,
              selected_weight=2., cv_weight=2., resume=False, stop_at=None, heartbeat):
    """Same draws, optimizer and schedule as native fit; only the loss changes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.pt'
    objective = dict(selected_weight=selected_weight, cv_weight=cv_weight)
    if any(not math.isfinite(v) or v < 0 for v in objective.values()):
        raise ValueError('Invalid loss weights')
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                           weight_decay=settings['weight_decay'])
    rng = torch.Generator().manual_seed(seed+7919)
    step, seconds, losses = 0, 0., []
    draws = np.zeros(len(data['sites']), np.int64)
    if checkpoint.exists():
        if not resume:
            raise FileExistsError('Existing checkpoint requires explicit --resume')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (saved['identity'] != identity or saved['settings'] != settings
                or saved['seed'] != seed or saved['objective'] != objective):
            raise ValueError('Checkpoint identity/settings/objective changed')
        np.testing.assert_array_equal(saved['train_ids'], fold['train_ids'])
        np.testing.assert_array_equal(saved['factors'], fold['factors'])
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        step, seconds, losses, draws = saved['step'], saved['seconds'], saved['losses'], saved['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or min(limit, settings['batch_size'], settings['checkpoint_every'], settings['heartbeat_every']) <= 0:
        raise ValueError('Invalid registered training/resume budget')
    allowed = np.zeros(len(data['sites']), bool)
    allowed[fold['train_ids']] = True
    model.train()

    def save():
        state = dict(identity=identity, settings=settings, seed=seed, objective=objective,
            step=step, seconds=seconds+time.monotonic()-started, losses=losses, draws=draws,
            train_ids=fold['train_ids'], factors=fold['factors'], model=model.state_dict(),
            optimizer=opt.state_dict(), sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state())
        temp = checkpoint.with_suffix('.tmp')
        torch.save(state, temp)
        os.replace(temp, checkpoint)

    try:
        while step < limit:
            ids = draw_batch(fold['groups'], settings['batch_size'], rng)
            if not allowed[ids].all():
                raise ValueError('Excluded source entered training')
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
            prediction = model(inputs)
            loss, raw_ade, components = reference_harm_objective(prediction, target, mask, factors,
                baseline_torch(inputs['history'], fold['baseline_index']),
                baseline_torch(inputs['history'], 1), **objective)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            step += 1
            np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), learning_rate=lr,
                    supported_batch_rows=int(mask.any(1).sum()), mean_past_normalized_ADE=float(raw_ade.detach().mean()))
                row.update({k:float(v.detach()) for k, v in components.items()})
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


def assert_training_equivalent(left, right):
    """Bitwise replay of scientific state, excluding runtime/identity/logging."""
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b
    for key in ('step', 'model', 'optimizer', 'sampler_rng', 'torch_rng', 'train_ids', 'factors', 'draws'):
        compare(left[key], right[key])
    return True
