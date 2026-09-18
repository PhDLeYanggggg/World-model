"""Fixed geometry-only conditioning/loss comparison, with exact phase resume."""
import os
import time

import numpy as np
import torch

from src.world_model.m3w_objective_alignment import geometry_prediction
from src.world_model.m3w_sdd_step_adapter import masked_future_ade

ARMS = ('unit_inputs_only', 'unit_primary_log', 'unit_internal_log')


def frame_prediction(model, batch, arm):
    if arm not in ARMS:
        raise ValueError('Unregistered conditioning arm')
    delta = geometry_prediction(model, batch['geometry'], batch['observed'], torch.zeros_like(batch['baseline']))
    rotated = torch.bmm(delta, batch['rotation'].transpose(1, 2))
    if arm != 'unit_inputs_only':
        rotated = rotated*batch['radius'][:, None, None]
    return batch['baseline']+torch.where(batch['spatial_support'][:, None, None], rotated, 0.)


def frame_loss(prediction, batch, arm):
    ade, valid = masked_future_ade(prediction, batch['target'], batch['valid'])
    supported = valid
    if arm == 'unit_internal_log':
        supported = valid & batch['spatial_support']
        scale = torch.where(batch['spatial_support'], batch['radius'], 1.)
        ade = ade/scale
    loss = ade[supported].log1p().mean() if supported.any() else prediction.sum()*0.
    return loss, dict(loss_rows=int(supported.sum()), valid_label_rows=int(valid.sum()),
        no_anchor_rows=int((~batch['spatial_support']).sum()),
        loss_coordinate='internal_observed_context' if arm=='unit_internal_log' else 'fixed_primary')


def fit_frame(model, main_batch, auxiliary_batch, counts, *, arm, config, seed,
              identity, checkpoint, heartbeat, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Fixed arm required')
    if len(counts) != 2 or min(counts) <= 0:
        raise ValueError('Nonempty main and auxiliary populations required')
    pre, total = config['pretraining_updates'], config['pretraining_updates']+config['main_updates']
    if pre <= 0 or total <= pre or config['batch_size'] <= 0 or config['checkpoint_every'] <= 0:
        raise ValueError('Positive fixed phase and checkpoint budgets required')
    if stop_at is not None and (not isinstance(stop_at, int) or stop_at < 1):
        raise ValueError('Pilot stopping step must be positive')
    limit = total if stop_at is None else min(total, stop_at)
    def optimizer():
        return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    opt = optimizer(); generator = torch.Generator().manual_seed(seed+7919)
    step, seconds, losses = 0, 0., []
    draws = [np.zeros(n, np.int64) for n in counts]
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['config'] != config or state['counts'] != counts or state['arm'] != arm):
            raise ValueError('Resume identity, budget or arm changed')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, losses, draws = state['step'], state['fit_seconds'], state['losses'], state['draw_counts']
    initial, start = step, time.monotonic()
    model.train()
    while step < limit:
        if step == pre:
            opt = optimizer(); generator.manual_seed(seed+104729)
        source = step < pre; group = 1 if source else 0
        ids = torch.randint(counts[group], (config['batch_size'],), generator=generator).numpy()
        batch = auxiliary_batch(ids) if source else main_batch(ids)
        opt.zero_grad(set_to_none=True)
        loss, detail = frame_loss(frame_prediction(model,batch,arm),batch,arm)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite frame loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step(); step += 1; np.add.at(draws[group],ids,1)
        if step == 1 or step % 100 == 0 or step == total:
            losses.append(dict(step=step,source=source,loss=float(loss.detach()),gradient_norm=float(gradient),**detail))
        if step in (1,pre,limit) or step % config['checkpoint_every'] == 0:
            elapsed = seconds+time.monotonic()-start
            state = dict(identity=identity,config=config,arm=arm,counts=counts,model=model.state_dict(),
                optimizer=opt.state_dict(),sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                step=step,fit_seconds=elapsed,losses=losses,draw_counts=draws)
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            tmp=checkpoint.with_suffix('.tmp'); torch.save(state,tmp); os.replace(tmp,checkpoint)
            heartbeat(state='fit_complete' if step==total else 'training',step=step,
                      fit_seconds=elapsed,loss=float(loss.detach()))
    model.eval()
    return dict(step=step,complete=step==total,new_updates_this_invocation=step-initial,
        fit_seconds=seconds+time.monotonic()-start,losses=losses,
        main_draws=int(draws[0].sum()),source_draws=int(draws[1].sum()))
