"""Continue frozen risk heads without using held outcomes to select an epoch."""
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.world_model import m3w_neural_risk_transport as parent

BUDGET = dict(minimum_extra_updates=4096, log_every=128, plateau_fraction=.01)


def training_loss(model, prepared):
    n = len(prepared['x'])
    total = 0.
    with torch.no_grad():
        for begin in range(0, n, 1024):
            sl = slice(begin, begin + 1024)
            total += float((prepared['w'][sl, None] *
                           (model(prepared['x'][sl]) - prepared['y'][sl]) ** 2).sum())
    return total / (2 * n)


def continue_fit(data, source, initial, identity, path, *, resume=False,
                 stop_extra_steps=None, heartbeat=lambda **kw: None, guard=lambda: None):
    p = parent.prepare(data, source)
    if not initial['complete'] or initial['settings'] != parent.SETTINGS:
        raise ValueError('Complete registered 32-epoch parent required')
    parent.storage.base.exact(initial['preprocess'], p['binding'])
    n = len(p['x'])
    if not (initial['visits'] == 32).all() or initial['cursor'] != n:
        raise ValueError('Parent must finish every fitting row at epoch 32')
    batches = math.ceil(n / parent.SETTINGS['batch_size'])
    extra_epochs = math.ceil(BUDGET['minimum_extra_updates'] / batches)
    target_steps = initial['step'] + extra_epochs * batches
    cfg = dict(parent.SETTINGS, epochs=32 + extra_epochs)
    path = Path(path)
    binding = dict(identity=identity, budget=BUDGET, settings=cfg,
                   preprocess=p['binding'], start_step=initial['step'], start_epoch=32,
                   target_step=target_steps, fit_rows=n, parameters=initial['parameters'])
    state = copy.deepcopy(initial)
    extra_trace = []
    elapsed = 0.
    if path.exists():
        if not resume:
            raise ValueError('Existing continuation requires explicit resume')
        state = parent.storage.base.read_checkpoint(path)
        for k, value in binding.items():
            parent.storage.base.exact(state[k], value)
        extra_trace, elapsed = state['extra_trace'], state['continuation_seconds']
    model = parent.Head(cfg)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    generator = torch.Generator()
    generator.set_state(state['generator'])
    torch.set_rng_state(state['torch_rng'])
    step, epoch, cursor, order, visits = (state[k] for k in
                                        ('step', 'epoch', 'cursor', 'order', 'visits'))
    limit = target_steps if stop_extra_steps is None else initial['step'] + stop_extra_steps
    if not step <= limit <= target_steps:
        raise ValueError('Continuation budget must be nondecreasing and within the fixed cap')
    started = time.monotonic()

    def log():
        row = dict(step=step, extra_steps=step-initial['step'], epoch=epoch,
                   weighted_fit_MSE=training_loss(model, p), minimum_visits=int(visits.min()))
        extra_trace.append(row)
        heartbeat(state='risk_budget_training', source=source, **row)

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        parent.storage.base.save_checkpoint(path, dict(
            **binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            generator=generator.get_state(), torch_rng=torch.get_rng_state(),
            step=step, epoch=epoch, cursor=cursor, order=order, visits=visits,
            extra_trace=extra_trace, continuation_seconds=elapsed+time.monotonic()-started,
            complete=step==target_steps))

    if not extra_trace:
        log()
    if not path.exists():
        save()
    try:
        while step < limit:
            guard()
            if cursor == n or not len(order):
                order = torch.randperm(n, generator=generator)
                cursor = 0
            rows = order[cursor:cursor+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = (p['w'][rows, None]*(model(p['x'][rows])-p['y'][rows])**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite continuation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            cursor += len(rows)
            visits[rows.numpy()] += 1
            if cursor == n:
                epoch += 1
            if (step-initial['step']) % BUDGET['log_every'] == 0 or step == target_steps:
                log()
                save()
            elif step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save()
        raise
    out = parent.storage.base.read_checkpoint(path)
    if out['complete'] and not (out['visits'] == cfg['epochs']).all():
        raise ValueError('The full continuation must finish complete epochs')
    return out


def exact_state(a, b):
    for k in ('identity', 'budget', 'settings', 'preprocess', 'start_step', 'target_step',
              'model', 'optimizer', 'generator', 'torch_rng', 'step', 'epoch', 'cursor',
              'order', 'visits', 'extra_trace', 'complete', 'parameters', 'fit_rows'):
        parent.storage.base.exact(a[k], b[k])


def fitting_diagnosis(state):
    trace = state['extra_trace']
    end = trace[-1]
    anchor = min(trace, key=lambda row: abs(row['extra_steps']-.75*end['extra_steps']))
    fraction = (anchor['weighted_fit_MSE']-end['weighted_fit_MSE'])/max(anchor['weighted_fit_MSE'], 1e-12)
    return dict(start_fit_loss=trace[0]['weighted_fit_MSE'], end_fit_loss=end['weighted_fit_MSE'],
                last_quarter_relative_reduction=fraction,
                operational_plateau=abs(fraction)<=BUDGET['plateau_fraction'],
                mathematical_convergence_proved=False, selected_by_held_outcomes=False)
