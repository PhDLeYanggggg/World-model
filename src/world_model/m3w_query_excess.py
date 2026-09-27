"""Matched query sampling for pointwise versus aggregate signed-risk training."""
from pathlib import Path
import platform
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
import numpy as np
import torch
from src.world_model import m3w_causal_descriptor_head as head
from scripts.replay_m3w_dimensionless_training import exact


def query_groups(sites, recordings, frames, known):
    sites, recordings, frames, known = map(np.asarray, (sites, recordings, frames, known))
    n = len(sites)
    if (any(a.shape != (n,) for a in (recordings, frames, known)) or known.dtype != bool):
        raise ValueError('Aligned source/recording/current-frame and supervision mask required')
    grouped = {}
    for i in np.flatnonzero(known):
        grouped.setdefault((str(sites[i]), str(recordings[i]), int(frames[i])), []).append(int(i))
    keys = sorted(grouped)
    groups = [np.asarray(grouped[k], np.int64) for k in keys]
    source_groups = [np.array([i for i, k in enumerate(keys) if k[0] == s], np.int64)
                     for s in sorted(set(sites))]
    if not groups or any(len(g) == 0 for g in source_groups):
        raise ValueError('Each fitting source needs an evaluable current query')
    return groups, source_groups, keys


def draw_queries(groups, source_groups, count, rng):
    if count <= 0 or count % len(source_groups):
        raise ValueError('Equal source allocation needs divisible query batch size')
    qids = np.concatenate([g[torch.randint(len(g), (count//len(source_groups),), generator=rng).numpy()]
                           for g in source_groups])
    blocks = [groups[i] for i in qids]
    return np.concatenate(blocks), np.repeat(np.arange(count), [len(x) for x in blocks]), qids


def losses(prediction, target, segments, count):
    if (prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 4
            or segments.shape != (len(target),) or segments.dtype != torch.int64
            or count <= 0 or not torch.isfinite(target).all() or (target < 0).any()
            or segments.min() < 0 or segments.max() >= count):
        raise ValueError('Known fitting costs and current-query segments required')
    error = head.parent.signed(prediction)-head.parent.signed(target.detach())
    size = torch.bincount(segments, minlength=count).to(error.dtype)
    if (size == 0).any():
        raise ValueError('Empty supervised query')
    total = error.new_zeros((count, 2)).index_add(0, segments, error)
    squared = error.new_zeros((count, 2)).index_add(0, segments, error.square())
    # Equal query weights in both arms isolate the aggregation objective.
    return {'pointwise': (squared/size[:, None]).mean(),
            'query': (total/size[:, None]).square().mean()}


def fit(x, u, y, sites, recordings, frames, envelope, pr, *, arm, seed, settings,
        identity, directory, heartbeat, resume=False, stop_at=None):
    if arm not in ('pointwise', 'query'):
        raise ValueError('Registered pointwise/query objective required')
    x, u, y, sites, recordings, frames, env = map(np.asarray,
        (x, u, y, sites, recordings, frames, envelope))
    known = pr['known']
    if (not np.array_equal(np.isfinite(y).all(1), known) or set(sites) != set(pr['training_sites'])
            or not np.isfinite(env).all() or (env < 0).any()
            or (y[known][:, [1, 3]] > env[known, None]+1e-5).any()):
        raise ValueError('Fitting-only costs, causal envelope and source roles required')
    groups, sg, keys = query_groups(sites, recordings, frames, known)
    upr = head.descriptor_preprocess(u, pr)
    mean_env = float(np.dot(pr['weights'], env))
    model = head.initialize(pr, settings['width'], seed, mean_env)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(np.clip((x.astype(float)-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    v = torch.from_numpy(np.clip((u.astype(float)-upr['mean'])/upr['std'], -upr['clip'], upr['clip']).astype(np.float32))
    targets = torch.from_numpy(np.where(known[:, None], y/pr['cost_scale'], 0).astype(np.float32))
    dt = torch.from_numpy((env/pr['cost_scale']).astype(np.float32))
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = draw_queries(groups, sg, 128, torch.Generator().manual_seed(seed+9137))
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt.gz'
    step = 0; elapsed = 0.; trace = []
    draws = np.zeros(len(x), np.int64); qdraws = np.zeros(len(groups), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing fit requires --resume')
        s = head.read_checkpoint(path)
        for key, value in dict(identity=identity, settings=settings, arm=arm, seed=seed,
            preprocess=pr, descriptor_preprocess=upr, initial_model=initial, mean_envelope=mean_env,
            query_keys=keys).items(): exact(s[key], value)
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, elapsed, trace, draws, qdraws = (s[k] for k in ('step', 'seconds', 'trace', 'draws', 'query_draws'))
    first = step; start = time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0: raise ValueError('Nondecreasing training budget required')
    def evaluate_monitor():
        ix, seg, _ = monitor
        with torch.no_grad():
            return {k: float(t) for k, t in losses(model(z[ix], dt[ix], v[ix]), targets[ix],
                torch.from_numpy(seg), 128).items()}
    if not trace: trace.append(dict(step=0, monitor=evaluate_monitor()))
    def save():
        head.save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, seed=seed,
            preprocess=pr, descriptor_preprocess=upr, mean_envelope=mean_env, initial_model=initial,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), query_keys=keys, step=step, seconds=elapsed+time.monotonic()-start,
            trace=trace, draws=draws, query_draws=qdraws))
    model.train()
    try:
        while step < limit:
            ix, seg, qids = draw_queries(groups, sg, settings['query_batch_size'], rng)
            opt.zero_grad(set_to_none=True)
            objective = losses(model(z[ix], dt[ix], v[ix]), targets[ix], torch.from_numpy(seg),
                               settings['query_batch_size'])[arm]
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite query-risk loss')
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1
            np.add.at(draws, ix, 1); np.add.at(qdraws, qids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
                row = dict(step=step, loss=float(objective.detach()), gradient_norm=float(norm), monitor=evaluate_monitor())
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_atomic_checkpoint', step=step); raise
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        parameters=sum(p.numel() for p in model.parameters()), supervised_queries=len(groups),
        singleton_queries=sum(len(g)==1 for g in groups), max_query_size=max(map(len, groups)),
        total_row_draws=int(draws.sum()), total_query_draws=int(qdraws.sum()),
        unknown_rows_sampled=int(draws[~known].sum()), trace=trace,
        seconds=elapsed+time.monotonic()-start, component_moments_identified=False)


def assert_matched(a, b):
    assert a['arm']=='pointwise' and b['arm']=='query'
    for k in ('settings', 'seed', 'preprocess', 'descriptor_preprocess', 'initial_model',
              'sampler_rng', 'torch_rng', 'draws', 'query_draws', 'query_keys', 'step', 'mean_envelope'):
        exact(a[k], b[k])


def quality(prediction, target, recordings, frames):
    known = np.isfinite(target).all(1)
    groups, _, _ = query_groups(np.repeat('held', len(target)), recordings, frames, known)
    error = head.parent.signed(np.asarray(prediction, float))-head.parent.signed(np.asarray(target, float))
    point = np.array([np.square(error[g]).mean(0) for g in groups]).mean(0)
    aggregate = np.array([np.square(error[g].mean(0)) for g in groups]).mean(0)
    return dict(pointwise_all_MSE=float(point[0]), pointwise_easy_MSE=float(point[1]),
        query_all_MSE=float(aggregate[0]), query_easy_MSE=float(aggregate[1]),
        queries=len(groups), singleton_fraction=float(np.mean([len(g)==1 for g in groups])))
