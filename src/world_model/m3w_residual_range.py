"""Matched residual readout/loss controls; inference has no target argument."""
from __future__ import annotations

import os
import time

import torch
from torch import nn

ARMS = ('linear_log', 'sinh_log', 'linear_asinh', 'sinh_asinh')


def decode_residual(raw, decoder, cap):
    if not 0 < cap <= 12:
        raise ValueError('Numerical cap must be in (0, 12]')
    if decoder == 'linear':
        return raw
    if decoder == 'sinh':
        return torch.sinh(raw.clamp(-cap, cap))
    raise ValueError('Unknown readout')


def range_prediction(model, features, observed, baseline, arm, cap, return_raw=False):
    if arm not in ARMS or observed.shape != (len(features), 8) or baseline.shape != (len(features), 12, 2):
        raise ValueError('Registered arm and complete eight/twelve inputs required')
    latent = features.new_zeros((len(features), 128))
    raw = model.output(model.fusion(torch.cat((features, observed, latent), 1))).reshape(-1, 12, 2)
    prediction = baseline + decode_residual(raw, arm.split('_')[0], cap)
    return (prediction, raw) if return_raw else prediction


def range_loss(prediction, target, baseline, objective):
    if prediction.shape != target.shape or target.shape != baseline.shape or target.shape[1:] != (12, 2):
        raise ValueError('Complete twelve-step labels required')
    ade = torch.linalg.vector_norm(prediction-target, dim=-1).mean(-1)
    reference = torch.linalg.vector_norm(baseline.detach()-target, dim=-1).mean(-1)
    if objective == 'log':
        loss = ade.log1p().mean()
    elif objective == 'asinh':
        loss = nn.functional.smooth_l1_loss(torch.asinh(prediction-baseline.detach()),
                                           torch.asinh(target-baseline.detach()), beta=1.)
    else:
        raise ValueError('Unknown objective')
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training objective')
    return loss, dict(ADE=float(ade.detach().mean()), CV_ADE=float(reference.detach().mean()),
                      positive_harm=float((ade-reference).clamp_min(0).detach().mean()))


def fit_range(model, batch, train_ids, *, arm, cap, config, seed, identity, checkpoint,
              heartbeat, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Unknown registered arm')
    weights = torch.ones(len(train_ids), dtype=torch.float64)/len(train_ids)
    generator = torch.Generator().manual_seed(seed+7919)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    step, losses, elapsed, saturated = 0, [], 0., 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['config'] != config or state['arm'] != arm
                or state['cap'] != cap or not torch.equal(state['train_ids'], train_ids)):
            raise ValueError('Resume identity/config/training membership changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        step, losses, elapsed, saturated = state['step'], state['losses'], state['fit_seconds'], state['saturated_coordinates']
    initial = step
    limit = config['updates'] if stop_at is None else min(stop_at, config['updates'])
    start = time.monotonic()
    model.train()
    while step < limit:
        chosen = torch.multinomial(weights, config['batch_size'], replacement=True, generator=generator)
        features, observed, baseline, target = batch(train_ids[chosen])
        optimizer.zero_grad(set_to_none=True)
        prediction, raw = range_prediction(model, features, observed, baseline, arm, cap, True)
        loss, detail = range_loss(prediction, target, baseline, arm.split('_')[1])
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        saturated += int((raw.detach().abs() >= cap).sum()) if arm.startswith('sinh') else 0
        if step == 1 or step % 100 == 0:
            losses.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), **detail))
        if step % config['checkpoint_every'] == 0 or step == limit:
            seconds = elapsed + time.monotonic()-start
            state = dict(identity=identity, config=config, arm=arm, cap=cap, train_ids=train_ids,
                model=model.state_dict(), optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(), step=step, losses=losses, fit_seconds=seconds,
                saturated_coordinates=saturated)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix('.tmp')
            torch.save(state, tmp)
            os.replace(tmp, checkpoint)
            heartbeat(dict(state='training' if step < config['updates'] else 'fit_complete',
                step=step, fit_seconds=seconds, loss=float(loss.detach()), saturated_coordinates=saturated))
    model.eval()
    return dict(step=step, complete=step == config['updates'], new_updates_this_invocation=step-initial,
                fit_seconds=elapsed+time.monotonic()-start, losses=losses, saturated_coordinates=saturated)
