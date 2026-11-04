"""Joint benefit/easy-harm/other-harm event learning on causal amplitude costs."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_bound_aware_cost as point

base = point.base
ARMS = ('product_moment', 'event_supervised')


class JointHarmHead(nn.Module):
    """Four exclusive events keep expected benefit plus harm below the envelope."""
    def __init__(self, dimension, width, seed, bias):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(dimension + 7, width)
        self.action = nn.Linear(width, 7)
        self.reference = nn.Linear(width, 2)
        with torch.no_grad():
            self.action.weight.zero_(); self.action.bias.zero_()
            self.reference.weight.zero_()
            self.reference.bias.copy_(torch.as_tensor(bias[[2, 3]]))

    def components(self, x, envelope, segments, count):
        if x.ndim != 3 or x.shape[1] != 7 or envelope.shape != x.shape[:2] or segments.shape != (len(x),):
            raise ValueError('Aligned complete causal queries required')
        code = torch.eye(7, dtype=x.dtype).expand(len(x), -1, -1)
        hidden = F.silu(self.encoder(torch.cat((x, code), -1)))
        membership = F.one_hot(segments, count).to(hidden.dtype).T
        total = membership @ hidden.flatten(1)
        sizes = membership.T @ membership.sum(1)
        others = ((membership.T @ total).reshape_as(hidden) - hidden)/(sizes-1).clamp_min(1)[:, None, None]
        hidden = F.silu(hidden + others)
        raw = self.action(hidden)
        logits, fractions = raw[..., :4], torch.sigmoid(raw[..., 4:])
        probability = F.softmax(logits, -1)
        # Categories: zero, benefit, non-easy harm, easy harm. No future input.
        mass = envelope[..., None]*probability[..., 1:]*fractions
        reference_raw = self.reference(hidden.mean(1))
        reference = F.softplus(reference_raw[:, 0])[:, None].expand(-1, 7)
        easy_reference = reference*torch.sigmoid(reference_raw[:, 1, None])
        benefit = torch.minimum(mass[..., 0], reference)
        harm = mass[..., 1] + mass[..., 2]
        moments = torch.stack((benefit, harm, reference, easy_reference, mass[..., 2]), -1)
        return dict(moments=moments, logits=logits, probability=probability, fractions=fractions)

    def forward(self, *args):
        return self.components(*args)['moments']


def event_targets(truth, envelope):
    """Labels only; caller must mask unobserved futures out of event losses."""
    event = torch.zeros(truth.shape[:2], dtype=torch.long)
    event[truth[..., 0] > 0] = 1
    event[truth[..., 1] > 0] = 2
    event[truth[..., 4] > 0] = 3
    fraction = torch.maximum(truth[..., 0], truth[..., 1])/envelope.clamp_min(1e-12)
    return event, fraction.clamp(0, 1)


def objective(parts, data, ix, segments, count, arm):
    if arm not in ARMS:
        raise ValueError('Fixed matched moment versus event-supervised arms required')
    result = point.objective(parts['moments'], data, ix, segments, count, 'bound_aware')
    truth = data['truth_by_arm']['bound_aware'][ix]
    c = data['causal']
    envelope = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
    event, fraction = event_targets(truth, envelope)
    known = torch.from_numpy(data['observed_future'][ix])
    legal = torch.from_numpy(c['allowed'][ix]) & known[:, None]
    positive = legal & (event > 0)
    membership = F.one_hot(segments, count).to(truth.dtype).T
    def average(value, mask):
        row = (value*mask).sum(1)/mask.sum(1).clamp_min(1)
        supported = mask.any(1)
        mass = membership @ supported.float()
        sums = membership @ (row*supported)
        return (sums/mass.clamp_min(1)).sum()/(mass > 0).sum().clamp_min(1)
    ce = F.cross_entropy(parts['logits'].flatten(0, 1), event.flatten(), reduction='none').reshape_as(event)
    severity = parts['fractions'].gather(-1, (event-1).clamp_min(0)[..., None]).squeeze(-1)
    occurrence = average(ce, legal)
    conditional = average((severity-fraction).square(), positive)
    auxiliary = occurrence + conditional
    return dict(result, total=result['total'] + (auxiliary if arm == 'event_supervised' else 0),
        occurrence=occurrence, conditional_fraction=conditional,
        event_supervised_rows=int(legal.any(1).sum()) if arm == 'event_supervised' else 0,
        event_positive_actions=int(positive.sum()), unknown_event_labels_used=0)


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Unregistered training arm')
    c, pr = data['causal'], data['preprocess']
    model = JointHarmHead(c['x'].shape[-1], settings['width'], seed, pr['bias'])
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm=arm, seed=seed, preprocess=pr,
        initial_model=initial, parameters=sum(p.numel() for p in model.parameters()), budget=budget,
        source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws = 0, 0., [], '0'*64
    if path.exists():
        if not resume: raise ValueError('Explicit unchanged resume required')
        old = base.read_checkpoint(path)
        for key, value in binding.items(): base.exact(old[key], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws = (old[k] for k in ('step', 'seconds', 'trace', 'draw_hash'))
    limit = budget if stop_at is None else min(budget, stop_at)
    if not 0 <= step <= limit or limit <= 0: raise ValueError('No budget rollback')
    started = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(),
            torch_rng=torch.get_rng_state(), trace=trace, draw_hash=draws, unknown_event_labels_used=0))
    if step == 0: save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
            envelope = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model.components(data['z'][ix], envelope, segments, len(qids)), data, ix, segments, len(qids), arm)
            if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite joint-event objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, gradient_norm=float(norm),
                    **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='joint_harm_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Every original fitting query must receive full coverage')
    return base.read_checkpoint(path)


def exact_state(a, b):
    for key in ('model', 'optimizer', 'sampler', 'torch_rng', 'step', 'trace', 'draw_hash',
                'fit_ids', 'source_sites', 'preprocess', 'initial_model', 'parameters', 'budget'):
        base.exact(a[key], b[key])


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Only declared causal inputs, never future targets or masks')
    pr = state['preprocess']
    model = JointHarmHead(causal['x'].shape[-1], state['settings']['width'], state['seed'], pr['bias'])
    model.load_state_dict(state['model']); model.eval()
    result = dict(moments=np.empty((len(causal['x']), 7, 5), float),
        probability=np.empty((len(causal['x']), 7, 4), float), fractions=np.empty((len(causal['x']), 7, 3), float))
    pending, size = [], 0
    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0
        scale = causal['scales'][ix]
        env = (causal['envelope'][ix]/scale[:, None]).astype(np.float32)
        with torch.no_grad():
            parts = model.components(torch.from_numpy(z), torch.from_numpy(env), torch.from_numpy(seg), len(pending))
        for key in result:
            v = parts[key].numpy().astype(float)
            result[key][ix] = v*scale[:, None, None] if key == 'moments' else v
    for group in causal['groups']:
        if pending and size+len(group) > 1024: flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending: flush()
    if not all(np.isfinite(v).all() and (v >= 0).all() for v in result.values()):
        raise FloatingPointError('Finite nonnegative cost outputs required')
    return result
