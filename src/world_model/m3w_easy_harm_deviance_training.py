"""Matched fixed-budget easy-harm loss comparison over unchanged causal heads."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from src.world_model import m3w_temporal_auxiliary as parent
from src.world_model import m3w_easy_harm_deviance as loss_api

ARMS = ('quadratic', 'easy_deviance')
core = parent.core


def fit(x, envelope, target, series, sites, recordings, frames, pr, *, arm,
        settings, seed, identity, experiment_sha256, path, heartbeat,
        resume=False, stop_at=None, checkpoint_guard=None):
    if checkpoint_guard:
        checkpoint_guard()
    if arm not in ARMS:
        raise ValueError('Registered paired loss required')
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
    groups, source_groups, keys = core.sampling.query_groups(sites, recordings, frames, known)
    z = torch.from_numpy(np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope/pr['scale']).astype(np.float32))
    truth = torch.from_numpy(np.where(known[:, None], target/pr['scale'], 0).astype(np.float32))
    aux_truth = torch.from_numpy((parent.auxiliary_targets(v, 'none')/pr['scale']).astype(np.float32))
    mask = torch.from_numpy(valid); rms = torch.from_numpy(pr['rms'])
    model = parent.initialize(pr, settings, seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = core.sampling.draw_queries(groups, source_groups, 128, torch.Generator().manual_seed(seed+9137))
    path = Path(path); step = 0; seconds = 0.; trace = []; draw_hash = '0'*64; draws = 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        old = core.read_checkpoint(path)
        for k, value in dict(identity=identity, experiment_sha256=experiment_sha256, settings=settings,
                             arm=arm, preprocess=pr, seed=seed, initial_model=initial, input_hashes=hashes).items():
            core.exact(old[k], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']); torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, draw_hash, draws = (old[k] for k in ('step','seconds','trace','draw_hash','row_draws'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing budget required')
    began = time.monotonic(); logits = []

    def capture(_module, _inputs, output):
        logits[:] = [output]

    hook = model.primary.output.register_forward_hook(capture)

    def terms(ix, seg, count):
        p, a = model(z[ix], env[ix]); segments = torch.from_numpy(seg)
        original = core.losses(p, truth[ix], segments, count, rms)
        primary = original if arm == 'quadratic' else loss_api.objective(
            p, truth[ix], loss_api.log_easy_harm(logits[0], env[ix]), env[ix], segments, count, rms)
        auxiliary = parent.temporal.auxiliary_loss(a, aux_truth[ix], mask[ix], segments, count)
        # Preserve the original zero-weight auxiliary graph and optimizer state.
        return dict(primary_total=primary['total'], moments=primary['moments'],
            decision_scores=primary['decision_scores'], auxiliary=auxiliary,
            total=primary['total']+0.*auxiliary, original_primary=original['total'])

    def monitor_loss():
        with torch.no_grad():
            return {k: float(v) for k, v in terms(*monitor[:2], 128).items()}

    def save():
        if checkpoint_guard:
            checkpoint_guard()
        path.parent.mkdir(parents=True, exist_ok=True)
        core.save_checkpoint(path, dict(identity=identity, experiment_sha256=experiment_sha256,
            settings=settings, arm=arm, preprocess=pr, seed=seed, initial_model=initial,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), step=step, trace=trace, seconds=seconds+time.monotonic()-began,
            draw_hash=draw_hash, row_draws=draws, queries=len(keys), unknown_rows_sampled=0,
            input_hashes=hashes, coefficient=0., primary_objective_unchanged=arm == 'quadratic'))

    try:
        if not trace:
            trace.append(dict(step=0, monitor=monitor_loss())); save()
        while step < limit:
            ix, seg, qids = core.sampling.draw_queries(groups, source_groups, settings['query_batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            terms_out = terms(ix, seg, settings['query_batch_size'])
            if not torch.isfinite(terms_out['total']):
                raise FloatingPointError('Nonfinite paired training loss')
            terms_out['total'].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; draws += len(ix)
            draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
                row = dict(step=step, loss=float(terms_out['total'].detach()), gradient_norm=float(gradient), monitor=monitor_loss())
                trace.append(row); heartbeat(state='training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
        return core.read_checkpoint(path)
    finally:
        hook.remove(); logits.clear()


def assert_matched(left, right):
    if (left['arm'], right['arm']) != ARMS:
        raise ValueError('Quadratic/deviance pair required')
    for k in ('identity','experiment_sha256','settings','preprocess','seed','step','initial_model',
              'input_hashes','sampler_rng','draw_hash','row_draws','queries'):
        core.exact(left[k], right[k])


def assert_original_control(new, original):
    if new['arm'] != 'quadratic' or original['arm'] != 'none':
        raise ValueError('Exact old no-auxiliary control required')
    for k in ('model','optimizer','initial_model','input_hashes','preprocess','settings','seed','step',
              'sampler_rng','torch_rng','draw_hash','row_draws','queries'):
        core.exact(new[k], original[k])


def gradient_probe(state, args):
    """A fixed TRAIN monitor probes decoder gradients without optimizer updates."""
    x, envelope, target, _, sites, rec, frames, pr = args
    known = np.isfinite(target).all(1)
    groups, sg, _ = core.sampling.query_groups(sites, rec, frames, known)
    ix, seg, _ = core.sampling.draw_queries(groups, sg, 128, torch.Generator().manual_seed(state['seed']+9137))
    z = torch.from_numpy(np.clip((x[ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope[ix]/pr['scale']).astype(np.float32))
    y = torch.from_numpy((target[ix]/pr['scale']).astype(np.float32)); rms = torch.from_numpy(pr['rms'])
    model = parent.initialize(pr, state['settings'], state['seed']); model.load_state_dict(state['model'])
    raw = model.primary.output(model.primary.encoder(z))
    logp = loss_api.log_easy_harm(raw, env)
    p = torch.where(env > 0, logp.exp(), torch.zeros_like(logp))
    square = ((p-y[:, 4])/rms[4]).square()
    deviance = loss_api.easy_deviance(logp, y[:, 4], env, rms[4])
    weights = 1/np.bincount(seg)[seg]/128
    w = torch.from_numpy(weights.astype(np.float32))
    gradients = [torch.autograd.grad((v*w).sum(), raw, retain_graph=True)[0][:, [1,4]] for v in (square, deviance)]
    if not all(torch.isfinite(v).all() for v in gradients):
        raise FloatingPointError('Nonfinite real TRAIN decoder gradient')
    return dict(rows=len(ix), positive_easy_harm_rows=int((y[:, 4] > 0).sum()),
        quadratic_logit_gradient_norm=float(gradients[0].norm()),
        deviance_logit_gradient_norm=float(gradients[1].norm()),
        scalar_loss_quadratic=float((square*w).sum().detach()),
        scalar_loss_deviance=float((deviance*w).sum().detach()),
        meaning='loss geometry diagnostic; norms are not directly comparable calibrated risks')
