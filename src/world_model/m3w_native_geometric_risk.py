"""Factor exact-reference risk into a learned event and a causal known distance."""
import os
from pathlib import Path
import platform
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import numpy as np
import torch
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized

ARMS = ('base_event', 'base_geometric', 'kinematic_event', 'kinematic_geometric')


def extend_features(x, geometry, arm):
    if arm not in ARMS or x.shape != (len(geometry), 357):
        raise ValueError('Registered causal feature arm required')
    g = np.asarray(geometry, float)
    xy, t = g[:, :16].reshape(-1, 8, 2), g[:, 16:24]
    dt = np.diff(t, axis=1)
    if not np.isfinite(g).all() or np.any(dt <= 0) or np.any(t > 0) or np.any(t[:, -1] != 0):
        raise ValueError('Strictly ordered past timestamps required')
    v = np.diff(xy, axis=1)/dt[..., None]
    dv = np.linalg.norm(v-v[:, -1:], axis=-1)
    acceleration = np.linalg.norm(np.diff(v, axis=1), axis=-1)/((dt[:, 1:]+dt[:, :-1])/2)
    backcast = xy[:, -1:] + t[..., None]*v[:, -1:]
    residual = np.linalg.norm(xy-backcast, axis=-1)
    extra = np.column_stack((dv.max(1), acceleration.mean(1), residual.mean(1), residual.max(1)))
    if arm.startswith('base_'):
        extra = np.zeros_like(extra)
    return np.column_stack((x, extra)).astype(np.float32)


def zero_targets(cv, full):
    cv, full = np.asarray(cv), np.asarray(full)
    if (cv.shape != full.shape or full.dtype != bool or not np.isfinite(cv[full]).all()
            or np.any(cv[full] < 0)):
        raise ValueError('Complete finite native baseline labels required')
    y = np.full(len(cv), np.nan)
    y[full] = cv[full] == 0
    return y


def build_head(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(361, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    with torch.no_grad():
        model[-1].weight.zero_(); model[-1].bias.zero_()
    return model


def objective(logit, target, distance, scale, arm):
    if arm not in ARMS or logit.shape != target.shape or distance.shape != target.shape:
        raise ValueError('Registered paired event objective required')
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
    cost = ((torch.sigmoid(logit)-target)*distance/scale).square().mean()
    return bce+(cost if arm.endswith('_geometric') else 0), bce, cost


def fit(x, y, distance, sites, pr, *, arm, seed, settings, identity, directory, resume=False, stop_at=None, heartbeat):
    model = build_head(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    target = torch.from_numpy(np.where(pr['known'], y, 0).astype(np.float32))
    dist = torch.from_numpy(np.asarray(distance, np.float32))
    if not np.array_equal(np.isfinite(y), pr['known']) or np.any(distance < 0) or not np.isfinite(distance).all():
        raise ValueError('Known training support and causal distance required')
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required for existing checkpoint')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if cp['identity'] != identity or cp['settings'] != settings or cp['arm'] != arm or cp['seed'] != seed:
            raise ValueError('Changed geometric risk resume identity')
        for k in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(pr[k], cp['preprocess'][k])
        assert pr['cost_scale'] == cp['preprocess']['cost_scale']
        model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng']); torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing registered budget required')
    def save():
        cp = dict(identity=identity, settings=settings, arm=arm, seed=seed, preprocess=pr,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), step=step, seconds=seconds+time.monotonic()-started,
            trace=trace, draws=draws)
        tmp = path.with_suffix('.tmp'); torch.save(cp, tmp); os.replace(tmp, path)
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            opt.zero_grad(set_to_none=True)
            loss, bce, cost = objective(model(z[ids])[:, 0], target[ids], dist[ids], pr['cost_scale'], arm)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite geometric loss')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                r = dict(step=step, loss=float(loss.detach()), bce=float(bce.detach()),
                    normalized_cost_mse=float(cost.detach()), gradient_norm=float(grad))
                trace.append(r); heartbeat(state='training', **r)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint', step=step); raise
    return dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unknown_rows_sampled=int(draws[~pr['known']].sum()), unique_training_rows=int((draws > 0).sum()),
        parameters=sum(p.numel() for p in model.parameters()), trace=trace)


def predict(model, x, pr):
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], pr))
            values.append(torch.sigmoid(model(z)[:, 0]).numpy())
    return np.concatenate(values)
