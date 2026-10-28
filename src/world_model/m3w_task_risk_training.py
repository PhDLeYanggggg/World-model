"""Matched cost heads with task-aligned utility and empirical risk duals."""
import copy
import hashlib
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')

import numpy as np
import torch

from src.world_model import m3w_easy_harm_deviance_training as old
from src.world_model import m3w_task_risk_objective as decision

ARMS = ('quadratic', 'task_utility', 'task_risk')
parent, core = old.parent, old.core


def validate_objective(config):
    if config != dict(task_weight=1., temperature_start=1., temperature_end=.1,
                      dual_learning_rate=1., dual_maximum=100., risk_budget=.02):
        raise ValueError('Registered decision objective required; no implicit tuning')


def temperature(step, settings, config):
    fraction = min(1., step/max(1, settings['steps']-1))
    return config['temperature_start']*(1-fraction)+config['temperature_end']*fraction


def fit(x, envelope, target, series, sites, recordings, frames, pr, *, eligible,
        arm, settings, objective_config, seed, identity, experiment_sha256,
        path, heartbeat, resume=False, stop_at=None, checkpoint_guard=None):
    if checkpoint_guard:
        checkpoint_guard()
    validate_objective(objective_config)
    if arm not in ARMS or eligible.shape != (len(x),) or eligible.dtype != bool:
        raise ValueError('Registered arm and causal eligibility required')
    core.exact(pr, core.preprocess(x, envelope, target, sites, recordings, frames,
                                  training_site=pr['training_site']))
    v, valid = parent.temporal.validate_series(series)
    known = np.isfinite(target).all(1)
    if len(v) != len(x) or not np.array_equal(known, valid.any(1)):
        raise ValueError('Unchanged supervision support required')
    means = np.where(valid[..., None], v, 0).sum(1)/valid.sum(1).clip(1)[:, None]
    np.testing.assert_allclose(means[known, 0], target[known, 1]-target[known, 0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(means[known, 1], target[known, 2], rtol=1e-7, atol=1e-7)
    hashes = {k: parent.forest.fingerprint(np.asarray(a)) for k, a in dict(x=x, envelope=envelope,
        primary=target, temporal=v, sites=sites, recordings=recordings, frames=frames).items()}
    eligible_hash = parent.forest.fingerprint(eligible)
    groups, source_groups, keys = core.sampling.query_groups(sites, recordings, frames, known)
    z = torch.from_numpy(np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope/pr['scale']).astype(np.float32))
    truth = torch.from_numpy(np.where(known[:, None], target/pr['scale'], 0).astype(np.float32))
    aux_truth = torch.from_numpy((parent.auxiliary_targets(v, 'none')/pr['scale']).astype(np.float32))
    mask, rms, eligibility = torch.from_numpy(valid), torch.from_numpy(pr['rms']), torch.from_numpy(eligible)
    model = parent.initialize(pr, settings, seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = core.sampling.draw_queries(groups, source_groups, 128, torch.Generator().manual_seed(seed+9137))
    path = Path(path); step = 0; seconds = 0.; trace = []; draw_hash = '0'*64; draws = 0
    dual = torch.zeros(2)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        state = core.read_checkpoint(path)
        for k, value in dict(identity=identity, experiment_sha256=experiment_sha256, settings=settings,
                             objective_config=objective_config, arm=arm, preprocess=pr, seed=seed,
                             initial_model=initial, input_hashes=hashes, eligibility_sha256=eligible_hash).items():
            core.exact(state[k], value)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draw_hash, draws, dual = (state[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws', 'dual'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing budget required')
    began = time.monotonic()

    def terms(ix, seg, count):
        p, a = model(z[ix], env[ix]); segments = torch.from_numpy(seg)
        primary = core.losses(p, truth[ix], segments, count, rms)
        auxiliary = parent.temporal.auxiliary_loss(a, aux_truth[ix], mask[ix], segments, count)
        t = temperature(step, settings, objective_config)
        task = decision.terms(p, truth[ix], segments, count, eligibility[ix], rms, t)
        total = primary['total']+0.*auxiliary
        if arm != 'quadratic':
            total = total+objective_config['task_weight']*decision.objective(task, dual)
        return dict(total=total, primary_total=primary['total'], moments=primary['moments'],
                    decision_scores=primary['decision_scores'], auxiliary=auxiliary,
                    decision_regret=task['regret'], soft_all_excess=task['risk_excess'][0],
                    soft_easy_excess=task['risk_excess'][1], soft_intervention=task['action_mean'])

    def monitor_loss():
        with torch.no_grad():
            return {k:float(v) for k,v in terms(*monitor[:2], 128).items()}

    def save():
        if checkpoint_guard:
            checkpoint_guard()
        path.parent.mkdir(parents=True, exist_ok=True)
        core.save_checkpoint(path, dict(identity=identity, experiment_sha256=experiment_sha256,
            settings=settings, objective_config=objective_config, arm=arm, preprocess=pr, seed=seed,
            initial_model=initial, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), dual=dual,
            step=step, trace=trace, seconds=seconds+time.monotonic()-began,
            draw_hash=draw_hash, row_draws=draws, queries=len(keys), unknown_rows_sampled=0,
            input_hashes=hashes, eligibility_sha256=eligible_hash, coefficient=0.,
            primary_objective_unchanged=arm == 'quadratic', risk_calibration_claim=False))

    if not trace:
        trace.append(dict(step=0, monitor=monitor_loss(), dual=dual.tolist())); save()
    while step < limit:
        ix, seg, qids = core.sampling.draw_queries(groups, source_groups, settings['query_batch_size'], rng)
        optimizer.zero_grad(set_to_none=True)
        loss = terms(ix, seg, settings['query_batch_size'])
        if not torch.isfinite(loss['total']):
            raise FloatingPointError('Nonfinite decision training loss')
        loss['total'].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        if arm == 'task_risk':
            dual = decision.update_dual(dual, torch.stack((loss['soft_all_excess'], loss['soft_easy_excess'])),
                objective_config['dual_learning_rate'], objective_config['dual_maximum'])
        step += 1; draws += len(ix)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(loss['total'].detach()), gradient_norm=float(gradient),
                       monitor=monitor_loss(), dual=dual.tolist())
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return core.read_checkpoint(path)


def assert_matched(states):
    if tuple(s['arm'] for s in states) != ARMS:
        raise ValueError('All three registered arms required')
    for other in states[1:]:
        for k in ('identity', 'experiment_sha256', 'settings', 'objective_config', 'preprocess',
                  'seed', 'step', 'initial_model', 'input_hashes', 'sampler_rng', 'draw_hash',
                  'row_draws', 'queries', 'eligibility_sha256'):
            core.exact(states[0][k], other[k])


def assert_original_control(state, control):
    old.assert_original_control(state, control)


predict = parent.predict
