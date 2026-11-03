"""Matched cost learning with observed moments versus explicit unknown bounds."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_selection_controller as base
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

ARMS = ('known_only', 'bound_aware')


def training_data(causal, rollouts, target, valid, floor_index, easy_cut):
    observed, known = base.labels(rollouts, target, valid, floor_index, easy_cut)
    common = base.training_data(causal, observed)
    bounded = supported_bound_targets(rollouts, target, valid, floor_index, easy_cut)
    np.testing.assert_array_equal(bounded['observed_future'], known)
    common['observed_future'] = known
    common['truth_by_arm'] = dict(known_only=common['truth'], bound_aware=torch.from_numpy(
        (bounded['moments']/causal['scales'][:, None, None]).astype(np.float32)))
    common['defined_by_arm'] = dict(known_only=known, bound_aware=bounded['target_defined'])
    return common


def objective(prediction, data, ix, segments, count, arm):
    if arm not in ARMS: raise ValueError('The fixed two-arm target comparison is required')
    defined = torch.from_numpy(data['defined_by_arm'][arm][ix])
    observed = torch.from_numpy(data['observed_future'][ix])
    legal = torch.from_numpy(data['causal']['allowed'][ix]) & defined[:, None]
    truth = data['truth_by_arm'][arm][ix]
    rms = torch.from_numpy(data['preprocess']['rms'])
    n = legal.sum(1).clamp_min(1)
    regression = (((prediction-truth)/rms).square().mean(-1)*legal).sum(1)/n
    delta = ((prediction[..., 0]-prediction[..., 1])-(truth[..., 0]-truth[..., 1]))/torch.linalg.vector_norm(rms[:2])
    gain = (delta.square()*legal).sum(1)/n
    membership = F.one_hot(segments, count).to(prediction.dtype).T
    supported = legal.any(1)
    def average(value):
        return ((membership @ (value*supported))/(membership @ supported.float()).clamp_min(1)).mean()
    reg, net = average(regression), average(gain)
    return dict(total=reg+.5*net, regression=reg, absolute_gain=net,
        observed_supervised_rows=int((supported & observed).sum()),
        unknown_bound_supervised_rows=int((supported & ~observed).sum()),
        unknown_rows_mislabeled_as_observed=0)


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_:None,
        guard=lambda:None, resume=False, stop_at=None):
    if arm not in ARMS: raise ValueError('A fixed registered target arm is required')
    c, pr = data['causal'], data['preprocess']
    model = base.CostHead(c['x'].shape[-1], settings['width'], seed, pr['bias'], 'context')
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm='context', target_arm=arm,
        seed=seed, preprocess=pr, initial_model=initial, parameters=sum(p.numel() for p in model.parameters()),
        budget=budget, source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed_draws, bound_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume: raise ValueError('Explicit same-binding resume required')
        old = base.read_checkpoint(path)
        for k,v in binding.items(): base.exact(old[k], v)
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed_draws, bound_draws = (
            old[k] for k in ('step','seconds','trace','draw_hash','observed_supervised_draws','unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0: raise ValueError('No budget rollback or zero-update result')
    start = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-start, trace=trace, draw_hash=draws,
            observed_supervised_draws=observed_draws, unknown_bound_supervised_draws=bound_draws,
            unknown_rows_mislabeled_as_observed=0))
    if step == 0: save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segment = torch.from_numpy(seg)
            envelope = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            opt.zero_grad(set_to_none=True)
            terms = objective(model(data['z'][ix], envelope, segment, len(qids)),data,ix,segment,len(qids),arm)
            if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite cost objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            opt.step(); step += 1
            observed_draws += terms['observed_supervised_rows']; bound_draws += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                record = dict(step=step,budget=budget,
                    **{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in terms.items()},
                    gradient_norm=float(norm),minimum_query_visits=int(sampler.counts.min()),
                    never_drawn_queries=int((sampler.counts==0).sum()))
                trace.append(record); heartbeat(state='bound_cost_training',target_arm=arm,**record)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    state = base.read_checkpoint(path)
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Every fitting query must receive its complete coverage budget')
    return state


def amplitudes(prediction, allowed, budget=.02, *, binary=False):
    p, legal = np.asarray(prediction), np.asarray(allowed)
    if (p.ndim != 3 or p.shape[1:] != (7,5) or legal.shape != p.shape[:2]
            or legal.dtype != bool or not np.isfinite(p).all() or (p<0).any() or budget != .02):
        raise ValueError('Finite seven-action moments and unchanged 2% budget required')
    b,h,r,er,eh = p[:,6].T
    alpha = np.ones(len(p),float)
    np.minimum(alpha, np.divide(budget*r,h,out=np.ones_like(h),where=h>0),out=alpha)
    np.minimum(alpha, np.divide(budget*er,eh,out=np.ones_like(eh),where=eh>0),out=alpha)
    take = legal[:,6] & (b>h) & (r>0) & (er>0)
    if binary: take &= alpha>=1
    alpha = np.where(take,alpha,0)
    return alpha


def prediction(floor, proposal, alpha):
    b,p,a = np.asarray(floor),np.asarray(proposal),np.asarray(alpha)
    if b.shape != p.shape or b.ndim != 3 or b.shape[-1] != 2 or a.shape != (len(b),):
        raise ValueError('One future-free amplitude per paired rollout required')
    if not all(np.isfinite(x).all() for x in (b,p,a)) or np.any((a<0)|(a>1)):
        raise ValueError('Finite bounded interpolation only')
    mixed = b+a[:,None,None]*(p-b)
    return np.where((a==1)[:,None,None],p,np.where((a==0)[:,None,None],b,mixed))
