"""Matched continuation with optional coherent cost-fraction link supervision."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from src.world_model import m3w_shared_motion_cost as original

base = original.base
ARMS = ('continue_original', 'decoder_auxiliary')


def auxiliary(raw, truth, envelope, allowed, segments, count):
    if (raw.shape != (*envelope.shape, 3) or truth.shape != (*envelope.shape, 5)
            or allowed.shape != envelope.shape or (envelope < 0).any()
            or not torch.isfinite(raw).all() or not torch.isfinite(truth).all()
            or not torch.isfinite(envelope).all() or (truth < 0).any()):
        raise ValueError('Finite coherent label moments and original decoder logits required')
    distance = torch.where(envelope > 0, envelope, 1)
    shares = truth[..., :2]/distance[..., None]
    if ((shares.sum(-1) > 1+2e-5) & allowed).any():
        raise ValueError('Label benefit/harm must fit original causal envelope')
    if (((envelope == 0) & (truth[..., :2].sum(-1) > 1e-7)) & allowed).any():
        raise ValueError('Zero-distance actions cannot have positive true benefit or harm')
    if ((truth[..., 4] > truth[..., 1]+1e-6) & allowed).any():
        raise ValueError('Easy harm cannot exceed all harm')
    distribution = torch.cat((shares, (1-shares.sum(-1)).clamp_min(0)[..., None]), -1).clamp_min(0)
    distribution = distribution/distribution.sum(-1, keepdim=True)
    logits = torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1)
    composition = F.kl_div(logits.log_softmax(-1), distribution.detach(), reduction='none').sum(-1)
    h = truth[..., 1]
    fraction = (truth[..., 4]/torch.where(h > 0, h, 1)).clamp(0, 1)
    easy = shares[..., 1]*F.binary_cross_entropy_with_logits(raw[..., 2], fraction.detach(), reduction='none')
    legal = allowed & (envelope > 0)
    per_row = ((composition+easy)*legal).sum(1)/legal.sum(1).clamp_min(1)
    membership = F.one_hot(segments, count).to(raw.dtype).T
    support = legal.any(1).to(raw.dtype)
    return ((membership @ (per_row*support))/(membership @ support).clamp_min(1)).mean()


def objective(prediction, raw, data, ix, segments, count, coefficient):
    if coefficient not in (0., 1.):
        raise ValueError('No post-readout auxiliary coefficient search')
    old = base.objective(prediction, data, ix, segments, count, 'bound_aware')
    env = torch.from_numpy((data['causal']['envelope'][ix]/data['causal']['scales'][ix, None]).astype(np.float32))
    legal = torch.from_numpy(data['causal']['allowed'][ix])
    extra = auxiliary(raw, data['truth_by_arm']['bound_aware'][ix], env, legal, segments, count)
    total = old['total'] if coefficient == 0 else old['total']+extra
    return dict(old, original_total=old['total'], decoder_auxiliary=extra, total=total)


def fit(data, parent, *, arm, settings, identity, path, guard=lambda:None,
        heartbeat=lambda **_:None, resume=False, stop_at=None):
    if arm not in ARMS or parent['step'] != 2000 or parent['identity']['feature_arm'] != 'local_history':
        raise ValueError('Fixed2k local-history parent and registered continuation arm required')
    data = original.for_arm(data, 'local_history')
    c, pr = data['causal'], data['preprocess']
    base.base.exact(pr, parent['preprocess'])
    base.base.exact(c['ids'], parent['fit_ids'])
    model = base.base.CostHead(original.DIMENSION, settings['width'], parent['seed'], pr['bias'], 'context')
    model.load_state_dict(parent['model'])
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    opt.load_state_dict(copy.deepcopy(parent['optimizer']))
    sampler = base.base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], parent['seed'])
    sampler.restore(copy.deepcopy(parent['sampler']))
    warm_counts = sampler.counts.copy()
    torch.set_rng_state(parent['torch_rng'].clone())
    binding = dict(identity=dict(identity, feature_arm='local_history'), arm=arm, settings=settings,
        seed=parent['seed'], preprocess=pr, initial_model=initial,
        parameters=sum(p.numel() for p in model.parameters()), budget=2000,
        fit_ids=c['ids'].copy(), parent_step=2000, parent_draw_hash=parent['draw_hash'],
        parent_query_counts=warm_counts, initial_optimizer=copy.deepcopy(opt.state_dict()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draws = 0, 0., [], '0'*64
    if path.exists():
        if not resume:
            raise ValueError('Explicit identical continuation resume required')
        saved = base.base.read_checkpoint(path)
        for key,value in binding.items():
            base.base.exact(saved[key], value)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        sampler.restore(saved['sampler'])
        torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, draws = (saved[k] for k in ('step', 'seconds', 'trace', 'draw_hash'))
    limit = 2000 if stop_at is None else min(2000, stop_at)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('No continuation budget rollback')
    started = time.monotonic()
    captured = []
    hook = model.action.register_forward_hook(lambda _, __, value:captured.append(value))

    def save():
        guard()
        base.base.save_checkpoint(path, dict(**binding, step=step, total_parent_plus_new_updates=2000+step,
            seconds=seconds+time.monotonic()-started, trace=trace, draw_hash=draws,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler=sampler.state(),
            torch_rng=torch.get_rng_state()))

    if step == 0:
        save()
    try:
        while step < limit:
            guard()
            ix, seg, queries = sampler.draw()
            segments = torch.from_numpy(seg)
            env = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            captured.clear()
            opt.zero_grad(set_to_none=True)
            prediction = model(data['z'][ix], env, segments, len(queries))
            if len(captured) != 1:
                raise ValueError('Original single action decode required')
            terms = objective(prediction, captured.pop(), data, ix, segments, len(queries), float(arm == 'decoder_auxiliary'))
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Nonfinite link continuation loss')
            terms['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            step += 1
            draws = hashlib.sha256(bytes.fromhex(draws)+queries.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == 2000:
                record = dict(step=step, budget=2000,
                    **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k,v in terms.items()},
                    gradient_norm=float(norm), additional_minimum_query_visits=int((sampler.counts-warm_counts).min()),
                    additional_never_drawn_queries=int(((sampler.counts-warm_counts) == 0).sum()))
                trace.append(record)
                heartbeat(state='warm_cost_continuation', arm=arm, **record)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save()
        raise
    finally:
        hook.remove()
    result = base.base.read_checkpoint(path)
    if step == 2000 and (sampler.counts-warm_counts).min() < 4:
        raise ValueError('Complete additional query coverage required')
    return result
