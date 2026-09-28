"""Single-factor auxiliary gradient norm cap for fixed easy-risk heads."""
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_easy_hurdle as api

np, torch = api.np, api.torch
ARMS = ('uncapped', 'risk_priority')
CAP = .5


def capped_gradient(risk, auxiliary, cap=CAP):
    if not 0 < cap < 1 or len(risk) != len(auxiliary) or not risk:
        raise ValueError('Aligned gradients and a strict priority cap required')
    if any(r.shape != a.shape for r, a in zip(risk, auxiliary)):
        raise ValueError('Gradient shapes differ')
    r = torch.cat([v.detach().flatten().double() for v in risk])
    a = torch.cat([v.detach().flatten().double() for v in auxiliary])
    if not torch.isfinite(r).all() or not torch.isfinite(a).all():
        raise FloatingPointError('Nonfinite component gradient')
    rn, an = float(r.norm()), float(a.norm())
    alpha = min(1., cap*rn/an) if an > 0 else 0.
    combined = tuple((rv.detach()+alpha*av.detach()) for rv, av in zip(risk, auxiliary))
    g = torch.cat([v.flatten().double() for v in combined])
    projection = float(torch.dot(r, g))/(rn*rn) if rn > 0 else None
    if projection is not None and projection < 1-cap-1e-6:
        raise FloatingPointError('Risk projection bound violated')
    return combined, dict(alpha=alpha, risk_gradient_norm=rn, auxiliary_gradient_norm=an,
                          risk_projection=projection, capped=bool(an > cap*rn))


def fit(x, u, target, sites, recordings, frames, envelope, pr, *, arm, seed,
        settings, identity, path, heartbeat, resume=False, stop_at=None):
    if arm == 'uncapped':
        return api.fit(x, u, target, sites, recordings, frames, envelope, pr,
            arm='supervised', seed=seed, settings=settings, identity=identity,
            path=path, heartbeat=heartbeat, resume=resume, stop_at=stop_at)
    if arm != 'risk_priority':
        raise ValueError('Unregistered arm')
    known = np.isfinite(target).all(1)
    if not np.array_equal(known, pr['known']) or set(sites) != set(pr['training_sites']):
        raise ValueError('Fitting-only role and known-label mask mismatch')
    if not (target[known, 0] == 1).any():
        raise ValueError('No fitting easy labels')
    z, env, norm = api.preprocess(x, u, envelope, pr)
    if (target[known, 2] > env.numpy()[known]+1e-5).any():
        raise ValueError('Observed harm exceeds the causal envelope')
    groups, sources, keys = api.sampling.query_groups(sites, recordings, frames, known)
    model = api.initialize(z.shape[1], settings['width'], seed)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    truth = torch.from_numpy(np.where(known[:, None], target, 0).astype(np.float32))
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = api.sampling.draw_queries(groups, sources, 128, torch.Generator().manual_seed(seed+9137))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, sample_hash = 0, 0., [], '0'*64
    query_draws, row_draws, cap_updates, alpha_sum = 0, 0, 0, 0.
    min_projection = None
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        s = api.head.read_checkpoint(path)
        for k, v in dict(identity=identity, settings=settings, arm=arm, seed=seed,
                         norm=norm, initial_model=initial, auxiliary_norm_cap=CAP).items():
            api.sampling.exact(s[k], v)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, seconds, trace, sample_hash, query_draws, row_draws = (s[k] for k in
            ('step', 'seconds', 'trace', 'sample_hash', 'query_draws', 'row_draws'))
        cap_updates, alpha_sum, min_projection = (s[k] for k in ('cap_updates', 'alpha_sum', 'min_risk_projection'))
    start = time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing training budget required')

    def monitor_loss():
        ix, seg, _ = monitor
        with torch.no_grad():
            return {k: float(v) for k, v in api.losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), 128).items()}

    if not trace:
        trace.append(dict(step=0, monitor=monitor_loss()))

    def save():
        api.head.save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, seed=seed,
            norm=norm, initial_model=initial, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=seconds+time.monotonic()-start, trace=trace, sample_hash=sample_hash,
            query_draws=query_draws, row_draws=row_draws, unknown_rows_sampled=0, fitting_queries=len(keys),
            auxiliary_norm_cap=CAP, cap_updates=cap_updates, alpha_sum=alpha_sum,
            min_risk_projection=min_projection, probability_calibration_certificate=False))

    params = tuple(model.parameters()); model.train()
    while step < limit:
        ix, seg, qids = api.sampling.draw_queries(groups, sources, settings['query_batch_size'], rng)
        optimizer.zero_grad(set_to_none=True)
        terms = api.losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), settings['query_batch_size'])
        if not all(torch.isfinite(v) for v in terms.values()):
            raise FloatingPointError('Nonfinite component loss')
        risk = torch.autograd.grad(terms['marginal'], params, retain_graph=True)
        auxiliary = torch.autograd.grad(terms['occurrence']+terms['conditional'], params)
        gradient, stats = capped_gradient(risk, auxiliary)
        for p, g in zip(params, gradient):
            p.grad = g
        grad = torch.nn.utils.clip_grad_norm_(params, settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        cap_updates += stats['capped']; alpha_sum += stats['alpha']
        if stats['risk_projection'] is not None:
            min_projection = stats['risk_projection'] if min_projection is None else min(min_projection, stats['risk_projection'])
        sample_hash = hashlib.sha256(bytes.fromhex(sample_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        query_draws += len(qids); row_draws += len(ix)
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, component_losses={k: float(v.detach()) for k, v in terms.items()},
                       gradient_norm=float(grad), gradient_control=stats, monitor=monitor_loss())
            trace.append(row); heartbeat(state='training', **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return api.head.read_checkpoint(path)


def assert_matched(a, b):
    for k in ('settings', 'seed', 'norm', 'initial_model', 'sampler_rng', 'torch_rng',
              'step', 'sample_hash', 'query_draws', 'row_draws', 'fitting_queries'):
        api.sampling.exact(a[k], b[k])
    assert a['arm'] == 'supervised' and b['arm'] == 'risk_priority'


def assert_parent_control(old, fresh):
    for k in old:
        if k not in ('seconds', 'identity'):
            api.sampling.exact(old[k], fresh[k])


def decisions(utility, risks, eligible, recordings, frames, ids, node_limit=256):
    if set(risks) != {'raw', *ARMS}:
        raise ValueError('Frozen raw risk and both trained arms required')
    anchors = {a: eligible & (q <= 0).all(1) for a, q in risks.items()}
    common = np.logical_and.reduce(list(anchors.values()))
    out, stats = dict(common_anchor=common), {}
    for arm, q in risks.items():
        if q.shape != (len(eligible), 2) or not np.isfinite(q).all():
            raise ValueError('Finite causal signed risk scores required')
        out[arm+'_independent'] = anchors[arm]
        for suffix, anchor in (('joint', anchors[arm]), ('matched', common)):
            actions, info = api.grouped(utility, q, eligible, anchor, recordings, frames, ids, node_limit=node_limit)
            out[arm+'_'+suffix] = actions['joint_utility']; stats[arm+'_'+suffix] = info
    return out, stats
