"""Causal subset supervision with a matched individual-error anchor."""
from pathlib import Path
import platform
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
import numpy as np
import torch
from src.world_model import m3w_query_excess as parent
head, exact = parent.head, parent.exact
ARMS = ('subset_pointwise', 'subset_aggregate')
SUBSETS = ('controller_admission', 'low_disagreement_half', 'high_disagreement_half')


def subset_masks(sites, recordings, frames, ids, moving, controller, envelope):
    sites, recordings, frames, ids, moving, controller, envelope = map(
        np.asarray, (sites, recordings, frames, ids, moving, controller, envelope))
    n = len(ids)
    if (any(x.shape != (n,) for x in (sites, recordings, frames, moving, controller, envelope))
            or moving.dtype != bool or controller.dtype != bool or len(set(ids)) != n
            or not np.isfinite(envelope).all() or (envelope < 0).any()):
        raise ValueError('Unique aligned IDs and finite causal subset inputs required')
    result = np.zeros((n, 3), bool)
    result[:, 0] = moving & controller
    groups = {}
    for i in range(n):
        groups.setdefault((str(sites[i]), str(recordings[i]), int(frames[i])), []).append(i)
    for group in groups.values():
        at = np.array([i for i in group if moving[i]], np.int64)
        order = at[np.lexsort((ids[at], envelope[at]))]
        cut = (len(at)+1)//2
        result[order[:cut], 1] = True
        result[order[cut:], 2] = True
    return result


def losses(prediction, target, segments, count, subsets):
    all_losses = parent.losses(prediction, target, segments, count)
    if subsets.shape != (len(target), 3) or subsets.dtype != torch.bool:
        raise ValueError('Three fixed causal subset masks required')
    error = head.parent.signed(prediction)-head.parent.signed(target.detach())
    w = subsets.to(error.dtype)
    size = error.new_zeros((count, 3)).index_add(0, segments, w)
    total = error.new_zeros((count, 3, 2)).index_add(0, segments, w[:, :, None]*error[:, None])
    squared = error.new_zeros((count, 3, 2)).index_add(0, segments, w[:, :, None]*error[:, None].square())
    # Empty subsets contribute zero in both arms; their coverage is reported separately.
    individual = (squared/size.clamp_min(1)[:, :, None]).mean()
    aggregate = (total/size.clamp_min(1)[:, :, None]).square().mean()
    return dict(subset_pointwise=.5*all_losses['pointwise']+.5*individual,
        subset_aggregate=.5*all_losses['pointwise']+.5*aggregate,
        anchor=all_losses['pointwise'], auxiliary_pointwise=individual,
        auxiliary_aggregate=aggregate, empty_subset_fraction=(size == 0).float().mean())


def fit(x, u, y, sites, recordings, frames, envelope, pr, subsets, *, arm, seed,
        settings, identity, directory, heartbeat, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Registered subset objective required')
    x, u, y, sites, recordings, frames, env, subsets = map(np.asarray,
        (x, u, y, sites, recordings, frames, envelope, subsets))
    known = pr['known']
    if (not np.array_equal(np.isfinite(y).all(1), known) or set(sites) != set(pr['training_sites'])
            or not np.isfinite(env).all() or (env < 0).any()
            or (y[known][:, [1, 3]] > env[known, None]+1e-5).any()
            or subsets.shape != (len(x), 3) or subsets.dtype != bool):
        raise ValueError('Fitting costs, causal envelope and fixed subsets required')
    groups, sg, keys = parent.query_groups(sites, recordings, frames, known)
    upr = head.descriptor_preprocess(u, pr); mean_env = float(np.dot(pr['weights'], env))
    model = head.initialize(pr, settings['width'], seed, mean_env)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(np.clip((x.astype(float)-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    v = torch.from_numpy(np.clip((u.astype(float)-upr['mean'])/upr['std'], -upr['clip'], upr['clip']).astype(np.float32))
    targets = torch.from_numpy(np.where(known[:, None], y/pr['cost_scale'], 0).astype(np.float32))
    dt = torch.from_numpy((env/pr['cost_scale']).astype(np.float32)); masks = torch.from_numpy(subsets)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = parent.draw_queries(groups, sg, 128, torch.Generator().manual_seed(seed+9137))
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt.gz'; step = 0; elapsed = 0.; trace = []
    draws = np.zeros(len(x), np.int64); qdraws = np.zeros(len(groups), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing fit requires --resume')
        s = head.read_checkpoint(path)
        for key, value in dict(identity=identity, settings=settings, arm=arm, seed=seed,
            preprocess=pr, descriptor_preprocess=upr, initial_model=initial, mean_envelope=mean_env,
            query_keys=keys, subsets=subsets).items(): exact(s[key], value)
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
                torch.from_numpy(seg), 128, masks[ix]).items()}
    if not trace: trace.append(dict(step=0, monitor=evaluate_monitor()))
    def save():
        head.save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, seed=seed,
            preprocess=pr, descriptor_preprocess=upr, mean_envelope=mean_env, initial_model=initial,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), query_keys=keys, subsets=subsets, step=step,
            seconds=elapsed+time.monotonic()-start, trace=trace, draws=draws, query_draws=qdraws))
    model.train()
    try:
        while step < limit:
            ix, seg, qids = parent.draw_queries(groups, sg, settings['query_batch_size'], rng)
            opt.zero_grad(set_to_none=True)
            objective = losses(model(z[ix], dt[ix], v[ix]), targets[ix], torch.from_numpy(seg),
                settings['query_batch_size'], masks[ix])[arm]
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite subset loss')
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1
            np.add.at(draws, ix, 1); np.add.at(qdraws, qids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
                row = dict(step=step, loss=float(objective.detach()), gradient_norm=float(norm), monitor=evaluate_monitor())
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step); raise
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        parameters=sum(p.numel() for p in model.parameters()), supervised_queries=len(groups),
        singleton_queries=sum(len(g)==1 for g in groups), max_query_size=max(map(len, groups)),
        total_row_draws=int(draws.sum()), total_query_draws=int(qdraws.sum()),
        unknown_rows_sampled=int(draws[~known].sum()), subset_selected_known_rows=subsets[known].sum(0).tolist(),
        subset_selected_unknown_rows=subsets[~known].sum(0).tolist(), trace=trace,
        seconds=elapsed+time.monotonic()-start, component_moments_identified=False)


def assert_matched(a, b):
    assert a['arm'] == ARMS[0] and b['arm'] == ARMS[1]
    for k in ('settings', 'seed', 'preprocess', 'descriptor_preprocess', 'initial_model',
              'sampler_rng', 'torch_rng', 'draws', 'query_draws', 'query_keys', 'step', 'mean_envelope', 'subsets'):
        exact(a[k], b[k])


def quality(prediction, target, recordings, frames, subsets):
    known = np.isfinite(target).all(1)
    groups, _, _ = parent.query_groups(np.repeat('held', len(target)), recordings, frames, known)
    error = head.parent.signed(np.asarray(prediction, float))-head.parent.signed(np.asarray(target, float))
    out = parent.quality(prediction, target, recordings, frames)
    for j, name in enumerate(SUBSETS):
        blocks = [g[subsets[g, j]] for g in groups]; nonempty = [g for g in blocks if len(g)]
        for i, metric in enumerate(('all', 'easy')):
            out[name+'_'+metric+'_MSE'] = float(np.mean([error[g, i].mean()**2 for g in nonempty])) if nonempty else None
        out[name+'_known_nonempty_fraction'] = len(nonempty)/len(blocks)
    return out
