"""Matched SOURCE cost training: current MSE, easy-only and dual harm deviance."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.world_model import m3w_selection_cardinal_cost as previous
from src.world_model.m3w_easy_harm_deviance import easy_deviance

base = previous.base
ARMS = ('mean_absolute', 'easy_deviance', 'dual_deviance')
NEW_ARMS = ARMS[1:]


def log_harms(raw, envelope):
    if (raw.ndim != 3 or raw.shape[1:] != (7, 3) or envelope.shape != raw.shape[:2]
            or not torch.isfinite(raw).all() or not torch.isfinite(envelope).all() or (envelope < 0).any()):
        raise ValueError('Existing action logits and causal envelopes required')
    log_fractions = F.log_softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
    safe = torch.where(envelope > 0, envelope, torch.ones_like(envelope))
    log_h = safe.log()+log_fractions[..., 1]
    return log_h, log_h+F.logsigmoid(raw[..., 2])


def objective(prediction, data, ix, segments, count, arm, raw=None, envelope=None):
    if arm not in ARMS:
        raise ValueError('Fixed matched cost loss arm required')
    original = previous.objective(prediction, data, ix, segments, count, 'mean_absolute')
    if arm == 'mean_absolute':
        return original
    logs = log_harms(raw, envelope)
    known = torch.from_numpy(data['known'][ix])
    legal = torch.from_numpy(data['causal']['allowed'][ix]) & known[:, None]
    target = data['truth'][ix]; rms = torch.from_numpy(data['preprocess']['rms'])
    square = ((prediction-target)/rms).square()
    dev = {}
    for channel, logp in zip((1, 4), logs):
        dev[channel] = easy_deviance(logp.flatten(), target[..., channel].flatten(),
            envelope.flatten(), rms[channel]).reshape_as(envelope)
    terms = torch.stack((square[..., 0], dev[1] if arm == 'dual_deviance' else square[..., 1],
                         square[..., 2], square[..., 3], dev[4]), -1)
    per_row = (terms.mean(-1)*legal).sum(1)/legal.sum(1).clamp_min(1)
    supported = legal.any(1)
    membership = F.one_hot(segments, count).to(prediction.dtype).T
    reg = ((membership @ (per_row*supported))/(membership @ supported.float()).clamp_min(1)).mean()
    return dict(total=reg+.5*original['ranking'], regression=reg, ranking=original['ranking'],
        original_total=original['total'], H_deviance=dev[1][legal].mean() if legal.any() else prediction.sum()*0,
        EH_deviance=dev[4][legal].mean() if legal.any() else prediction.sum()*0,
        unknown_supervised_rows=original['unknown_supervised_rows'])


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_:None,
        guard=lambda:None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Fixed mean/easy/dual comparison required')
    c, pr = data['causal'], data['preprocess']
    model = base.CostHead(c['x'].shape[-1], settings['width'], seed, pr['bias'], 'context')
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm='context', objective_arm=arm, seed=seed,
        preprocess=pr, initial_model=initial, parameters=sum(p.numel() for p in model.parameters()),
        budget=budget, source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, unknown = 0, 0., [], '0'*64, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged-definition resume required')
        old = base.read_checkpoint(path)
        for key, value in binding.items(): base.exact(old[key], value)
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, unknown = (old[k] for k in ('step', 'seconds', 'trace', 'draw_hash', 'unknown_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('Nondecreasing complete-coverage budget required')
    started = time.monotonic(); captured = []
    def capture(_module, _inputs, output):
        captured[:] = [output]
    hook = model.action.register_forward_hook(capture) if arm != 'mean_absolute' else None

    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, draw_hash=draws,
            unknown_draws=unknown, unknown_supervised_rows=0))

    if step == 0: save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segment = torch.from_numpy(seg)
            env = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            opt.zero_grad(set_to_none=True)
            prediction = model(data['z'][ix], env, segment, len(qids))
            terms = objective(prediction, data, ix, segment, len(qids), arm,
                              captured[0] if captured else None, env)
            if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']:
                raise FloatingPointError('Finite known-label-only loss required')
            terms['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            unknown += int((~data['known'][ix]).sum())
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                record = dict(step=step, budget=budget,
                    **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k,v in terms.items()},
                    gradient_norm=float(norm), minimum_query_visits=int(sampler.counts.min()),
                    never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(record); heartbeat(state='harm_deviance_training', objective_arm=arm, **record)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    finally:
        if hook is not None: hook.remove()
        captured.clear()
    state = base.read_checkpoint(path)
    if step == budget and int(sampler.counts.min()) < settings['largest_source_cycles']:
        raise AssertionError('Every original SOURCE query requires full registered coverage')
    return state
