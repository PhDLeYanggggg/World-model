"""Preserve uniform-row ADE risk under the frozen episode sampler."""
import time

import numpy as np
import torch

from src.world_model.m3w_source_continuation import atomic_checkpoint, learning_rate
from src.world_model.m3w_source_cost_dynamics import restore

OBJECTIVE = 'uniform_row_ADE_exact_inverse_probability_no_self_normalization'


def uniform_risk_factors(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 1 or not len(p) or not np.isfinite(p).all()
            or np.any(p <= 0) or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Strictly positive normalized full-training probabilities required')
    factors = p.sum() / (len(p) * p)
    if not np.isfinite(factors).all():
        raise ValueError('Nonfinite inverse-probability factor')
    return factors


def importance_objective(prediction, target, scale, factors):
    if (prediction.shape != target.shape or target.ndim != 3 or target.shape[1:] != (12, 2)
            or not len(target) or not np.isfinite(scale) or scale <= 0
            or factors.shape != (len(target),) or not torch.isfinite(factors).all()
            or torch.any(factors <= 0) or not torch.isfinite(target).all()):
        raise ValueError('Complete targets, positive training scale and row factors required')
    ade = torch.linalg.vector_norm(prediction - target, dim=-1).mean(1) / scale
    # Do not divide by sampled factor sum: that would change the estimator.
    loss = (ade * factors).mean()
    return loss, ade.mean(), torch.any(target != 0, dim=(1, 2)).sum()


def fit_importance_candidate(model, inputs, targets, ids, probabilities, *, scale,
                             seed, config, identity, checkpoint, heartbeat, stop_at=None):
    ids = np.asarray(ids, dtype=int)
    p = np.asarray(probabilities, dtype=np.float64)
    correction = uniform_risk_factors(p)
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids) or p.shape != ids.shape
            or config['start_step'] <= 0 or config['updates'] <= config['start_step'] + 1
            or min(config['checkpoint_every'], config['batch_size']) <= 0):
        raise ValueError('Fixed full-population sampler and two-phase budget required')
    weights = torch.as_tensor(p, dtype=torch.float64)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = torch.Generator().manual_seed(seed + 7919)
    step, seconds, trace = 0, 0., []
    draws = np.zeros(len(ids), dtype=np.int64)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (saved['identity'] != identity or saved['config'] != config or saved['scale'] != scale
                or saved['objective'] != OBJECTIVE):
            raise ValueError('Checkpoint provenance or objective changed')
        np.testing.assert_array_equal(saved['train_ids'], ids)
        np.testing.assert_array_equal(saved['probabilities'], p)
        np.testing.assert_array_equal(saved['importance_factors'], correction)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, draws = saved['step'], saved['fit_seconds'], saved['trace'], saved['draw_counts']
    first, started = step, time.monotonic()
    limit = config['updates'] if stop_at is None else min(config['updates'], stop_at)
    if limit <= 0 or step > limit:
        raise ValueError('Invalid pilot or saved step')
    model.train()
    try:
        while step < limit:
            rate = config['learning_rate'] if step < config['start_step'] else learning_rate('cosine', step, config)
            for group in opt.param_groups:
                group['lr'] = rate
            local = torch.multinomial(weights, config['batch_size'], replacement=True, generator=rng).numpy()
            batch = ids[local]
            features, frame = inputs(batch); target = targets(batch)
            factors = torch.as_tensor(correction[local], dtype=target.dtype, device=target.device)
            opt.zero_grad(set_to_none=True)
            prediction = restore(model(*features, 'mask_only'), *frame)
            loss, all_ade, moving = importance_objective(prediction, target, scale, factors)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite importance-corrected objective')
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
            opt.step(); step += 1; np.add.at(draws, local, 1)
            if step == 1 or step % 100 == 0:
                trace.append(dict(step=step, learning_rate=rate, objective_loss=float(loss.detach()),
                    uncorrected_batch_ade=float(all_ade.detach()), nonzero_target_rows=int(moving),
                    gradient_norm=grad, sampled_factor_mean=float(factors.mean()),
                    sampled_factor_min=float(factors.min()), sampled_factor_max=float(factors.max())))
            if step == limit or step % config['checkpoint_every'] == 0:
                saved = dict(identity=identity, config=config, scale=scale, objective=OBJECTIVE,
                    probabilities=p, importance_factors=correction,
                    model=model.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
                    torch_rng=torch.get_rng_state(), step=step, fit_seconds=seconds+time.monotonic()-started,
                    trace=trace, train_ids=ids, draw_counts=draws)
                atomic_checkpoint(checkpoint, saved)
                heartbeat(state='training', step=step, objective_loss=float(loss.detach()),
                          uncorrected_batch_ade=float(all_ade.detach()), fit_seconds=saved['fit_seconds'])
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_completed_step=step)
        raise
    model.eval()
    return dict(step=step, complete=step == config['updates'], new_updates=step-first,
        fit_seconds=seconds+time.monotonic()-started, trace=trace,
        total_draws=int(draws.sum()), distinct_sampled_rows=int((draws > 0).sum()))
