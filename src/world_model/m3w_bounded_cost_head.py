"""Matched direct and forecast-disagreement-bounded continuous cost heads."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized

ARMS = ('direct_native', 'bounded_native', 'bounded_fraction')


class CostHead(torch.nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(features, width), torch.nn.GELU(), torch.nn.Linear(width, 2))

    def forward(self, x, scaled_distance, arm):
        if arm not in ARMS or scaled_distance.shape != (len(x),):
            raise ValueError('Fixed cost arm and past-only disagreement required')
        positive = torch.nn.functional.softplus(self.network(x))
        if arm == 'direct_native':
            score = positive
        else:
            score = scaled_distance[:, None]*positive/(1+positive.sum(1, keepdim=True))
        return torch.where(scaled_distance[:, None] == 0, 0., score)


def build(features, width, seed):
    torch.manual_seed(seed)
    model = CostHead(features, width)
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.zero_()
    return model


def loss_value(score, target, scaled_distance, arm):
    if (arm not in ARMS or score.shape != target.shape or target.shape != (len(score), 2)
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Supported continuous benefit/harm supervision required')
    if arm == 'bounded_fraction':
        denominator = torch.where(scaled_distance > 0, scaled_distance, 1.)[:, None]
        return ((score-target.detach())/denominator).square().mean()
    return (score-target.detach()).square().mean()


def predict(model, x, distance, pr, arm):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], pr))
            d = torch.from_numpy((distance[start:start+4096]/pr['cost_scale']).astype(np.float32))
            out.append(model(z, d, arm).numpy().astype(float)*pr['cost_scale'])
    return np.concatenate(out)


def fit(x, y, distance, sites, pr, *, arm, seed, settings, identity, directory, heartbeat, resume=False, stop_at=None):
    model = build(x.shape[1], settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    d = torch.from_numpy((distance/pr['cost_scale']).astype(np.float32))
    target = torch.from_numpy(np.where(pr['known'][:, None], y/pr['cost_scale'], 0).astype(np.float32))
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    path = Path(directory)/'checkpoint.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if cp['identity'] != identity or cp['settings'] != settings or cp['arm'] != arm or cp['seed'] != seed:
            raise ValueError('Changed resume identity')
        for k in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(cp['preprocess'][k], pr[k])
        assert cp['preprocess']['cost_scale'] == pr['cost_scale']
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng'])
        torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if not 0 < limit or limit < step:
        raise ValueError('Nondecreasing fixed training budget required')
    def save():
        state = dict(identity=identity, settings=settings, arm=arm, seed=seed, preprocess=pr,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), step=step, seconds=seconds+time.monotonic()-started, trace=trace, draws=draws)
        temp = path.with_suffix('.tmp')
        torch.save(state, temp)
        os.replace(temp, path)
    try:
        model.train()
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            score = model(z[ids], d[ids], arm)
            loss = loss_value(score, target[ids], d[ids], arm)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite bounded cost loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row)
                heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step)
        raise
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unknown_rows_sampled=int(draws[~pr['known']].sum()), trace=trace,
        parameters=sum(p.numel() for p in model.parameters()))
