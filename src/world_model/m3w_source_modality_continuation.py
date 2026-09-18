"""Explicit-modality version; the registered RGB predecessor stays immutable."""
import copy
import time

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_source_continuation import SCHEDULES, atomic_checkpoint, learning_rate
from src.world_model.m3w_source_cost_dynamics import dynamics_loss, restore


def continue_modality(model, inputs, targets, train_ids, weights, *, parent,
                 arm, schedule, config, identity, checkpoint, heartbeat,
                 snapshot, stop_at=None):
    ids = np.asarray(train_ids, dtype=int)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    if (arm not in ('mask_only', 'past_rgb') or schedule not in SCHEDULES or len(np.unique(ids)) != len(ids)
            or weights.shape != (len(ids),) or not len(ids)
            or torch.any(weights <= 0) or not torch.isclose(weights.sum(), torch.tensor(1., dtype=weights.dtype))
            or config['milestones'][0] != config['start_step']
            or config['milestones'][-1] != config['updates']
            or config['milestones'] != sorted(set(config['milestones']))
            or config['checkpoint_every'] <= 0):
        raise ValueError('Fixed aligned training complement and milestones required')
    if (parent['step'] != config['start_step'] or parent['arm'] != arm
            or parent['objective'] != 'ade' or parent['normalizer'] <= 0):
        raise ValueError('Verified ADE/modality parent checkpoint required')
    np.testing.assert_array_equal(parent['train_ids'], ids)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    generator = torch.Generator()
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity or state['config'] != config or state['schedule'] != schedule or state['arm'] != arm:
            raise ValueError('Continuation identity/config/schedule changed')
        np.testing.assert_array_equal(state['train_ids'], ids)
        step, seconds = state['step'], state['continuation_seconds']
        trace, clipped, maximum_grad = state['trace'], state['clipped_updates'], state['maximum_gradient_norm']
    else:
        state = parent
        step, seconds, trace, clipped, maximum_grad = config['start_step'], 0., [], 0, 0.
    if state['normalizer'] != parent['normalizer']:
        raise ValueError('Parent loss scale changed')
    model.load_state_dict(state['model'])
    # Optimizer loading can retain CPU tensor storage; never mutate a parent fork.
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    generator.set_state(state['sampler_rng'])
    torch.set_rng_state(state['torch_rng'])
    draws = state['draw_counts'].copy()
    first, started = step, time.monotonic()
    limit = config['updates'] if stop_at is None else min(stop_at, config['updates'])
    if limit < config['start_step'] or step > config['updates']:
        raise ValueError('Invalid continuation budget')

    def payload():
        return dict(identity=identity, config=config, schedule=schedule,
                    normalizer=parent['normalizer'], arm=arm, objective='ade',
                    model=model.state_dict(), optimizer=opt.state_dict(),
                    sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                    step=step, continuation_seconds=seconds + time.monotonic() - started,
                    trace=trace, clipped_updates=clipped, maximum_gradient_norm=maximum_grad,
                    train_ids=ids, draw_counts=draws)

    if not checkpoint.exists():
        atomic_checkpoint(checkpoint, payload())
    if step in config['milestones']:
        snapshot(torch.load(checkpoint, map_location='cpu', weights_only=False))
    model.train()
    try:
        while step < limit:
            rate = learning_rate(schedule, step, config)
            for group in opt.param_groups:
                group['lr'] = rate
            local = torch.multinomial(weights, config['batch_size'], replacement=True,
                                      generator=generator).numpy()
            ids_batch = ids[local]
            features, frame = inputs(ids_batch)
            target = targets(ids_batch)
            opt.zero_grad(set_to_none=True)
            prediction = restore(model(*features, arm), *frame)
            loss, ade = dynamics_loss(prediction, target, parent['normalizer'], 'ade')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite continuation loss')
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
            opt.step()
            step += 1
            np.add.at(draws, local, 1)
            clipped += int(grad > 5.)
            maximum_grad = max(maximum_grad, grad)
            if step == config['start_step'] + 1 or step % 100 == 0:
                trace.append(dict(step=step, learning_rate=rate,
                                  normalized_batch_ade=float(ade.detach()), gradient_norm=grad))
            if step == limit or step % config['checkpoint_every'] == 0 or step in config['milestones']:
                state = payload()
                atomic_checkpoint(checkpoint, state)
                heartbeat(state='training', step=step, learning_rate=rate,
                          normalized_batch_ade=float(ade.detach()),
                          continuation_seconds=state['continuation_seconds'])
                if step in config['milestones']:
                    snapshot(state)
                    model.train()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_completed_step=step)
        raise
    model.eval()
    return dict(step=step, complete=step == config['updates'], new_updates=step-first,
                additional_updates=step-config['start_step'],
                continuation_seconds=seconds+time.monotonic()-started,
                clipped_updates=clipped, maximum_gradient_norm=maximum_grad,
                total_draws=int(draws.sum()), mean_draws_per_row=float(draws.mean()),
                distinct_sampled_rows=int((draws > 0).sum()), trace=trace)
