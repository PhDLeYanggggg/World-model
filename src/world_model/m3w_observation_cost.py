"""Observation-support factorization for the existing masked-cost contract."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_selection_controller as base
from src.world_model import m3w_bound_aware_cost as direct

ARMS = ('marginal_support', 'identified_support')


def compose(probability, conditional, radius):
    if (conditional.ndim != 3 or conditional.shape[1:] != (7, 5)
            or probability.shape != (len(conditional),) or radius.shape != conditional.shape[:2]):
        raise ValueError('One observation probability and seven five-moment actions per row required')
    unknown = torch.stack((torch.zeros_like(radius), radius, torch.zeros_like(radius),
        torch.zeros_like(radius), radius), -1)
    q = probability[:, None, None]
    return q*conditional+(1-q)*unknown


class ObservationCostHead(base.CostHead):
    def __init__(self, dimension, width, seed, bias, observation_prior):
        if not 0 < observation_prior < 1:
            raise ValueError('Finite fixed observation prior required')
        super().__init__(dimension, width, seed, bias, 'context')
        self.support = nn.Linear(width, 1)
        with torch.no_grad():
            self.support.weight.zero_()
            self.support.bias.fill_(float(np.log(observation_prior/(1-observation_prior))))

    def forward(self, x, envelope, radius, segments, count):
        if (x.ndim != 3 or x.shape[1] != 7 or envelope.shape != x.shape[:2]
                or radius.shape != x.shape[:2] or segments.shape != (len(x),)):
            raise ValueError('Complete causal query packets and causal radii required')
        code = torch.eye(7, dtype=x.dtype).expand(len(x), -1, -1)
        hidden = F.silu(self.encoder(torch.cat((x, code), -1)))
        membership = F.one_hot(segments, count).to(hidden.dtype).T
        total = membership @ hidden.flatten(1)
        sizes = membership.T @ membership.sum(1)
        others = ((membership.T @ total).reshape_as(hidden)-hidden)/(sizes-1).clamp_min(1)[:, None, None]
        hidden = F.silu(hidden+others)
        raw = self.action(hidden); shared = hidden.mean(1); ref = self.reference(shared)
        reference = F.softplus(ref[:, 0])[:, None].expand(-1, 7)
        easy_reference = reference*torch.sigmoid(ref[:, 1, None])
        fraction = F.softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
        benefit = torch.minimum(envelope*fraction[..., 0], reference)
        harm = envelope*fraction[..., 1]
        conditional = torch.stack((benefit, harm, reference, easy_reference, harm*torch.sigmoid(raw[..., 2])), -1)
        logit = self.support(shared)[:, 0]; probability = torch.sigmoid(logit)
        return dict(moments=compose(probability, conditional, radius), conditional=conditional,
            observation_logit=logit, observation_probability=probability)


def prepare(learning, paths, floor_index):
    radius = np.linalg.norm(paths-paths[:, floor_index, None], axis=-1).max(-1)
    if radius.shape != learning['causal']['envelope'].shape or not np.isfinite(radius).all():
        raise ValueError('Finite candidate/floor causal disagreement required')
    return dict(learning, radius=(radius/learning['causal']['scales'][:, None]).astype(np.float32))


def objective(parts, data, ix, segments, count, arm):
    if arm not in ARMS: raise ValueError('Registered matched factorization arms required')
    terms = direct.objective(parts['moments'], data, ix, segments, count, 'bound_aware')
    legal = torch.from_numpy(data['causal']['allowed'][ix])
    observed = torch.from_numpy(data['observed_future'][ix])
    membership = F.one_hot(segments, count).to(parts['moments'].dtype).T
    def average(value, supported):
        return ((membership @ (value*supported))/(membership @ supported.float()).clamp_min(1)).mean()
    bce = average(F.binary_cross_entropy_with_logits(parts['observation_logit'], observed.float(), reduction='none'), legal.any(1))
    supported = legal & observed[:, None]
    rms = torch.from_numpy(data['preprocess']['rms'])
    error = ((parts['conditional']-data['truth'][ix])/rms).square().mean(-1)
    row_error = (error*supported).sum(1)/supported.sum(1).clamp_min(1)
    conditional = average(row_error, supported.any(1))
    anchor = terms['total']; auxiliary = bce+conditional
    terms.update(total=anchor+(auxiliary if arm == 'identified_support' else 0*auxiliary),
        bound_anchor=anchor, observation_BCE=bce, observed_conditional_MSE=conditional,
        identification_auxiliary=auxiliary)
    return terms


def fit(data, *, arm, seed, settings, observation_prior, identity, path,
        heartbeat=lambda **_: None, guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS: raise ValueError('Fixed observation-factorization arm required')
    c, pr = data['causal'], data['preprocess']
    model = ObservationCostHead(c['x'].shape[-1], settings['width'], seed, pr['bias'], observation_prior)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, target_arm=arm, observation_prior=observation_prior,
        seed=seed, preprocess=pr, initial_model=initial, parameters=sum(p.numel() for p in model.parameters()),
        budget=budget, source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed_draws, bound_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume: raise ValueError('Explicit unchanged-definition resume required')
        old = base.read_checkpoint(path)
        for k, v in binding.items(): base.exact(old[k], v)
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed_draws, bound_draws = (
            old[k] for k in ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0: raise ValueError('No budget rollback or zero-update completion')
    start = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step, seconds=elapsed+time.monotonic()-start,
            trace=trace, draw_hash=draws, observed_supervised_draws=observed_draws,
            unknown_bound_supervised_draws=bound_draws, unknown_rows_mislabeled_as_observed=0))
    if step == 0: save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
            env = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            radius = torch.from_numpy(data['radius'][ix]); opt.zero_grad(set_to_none=True)
            terms = objective(model(data['z'][ix], env, radius, segments, len(qids)), data, ix, segments, len(qids), arm)
            if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite support-factorized objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1
            observed_draws += terms['observed_supervised_rows']; bound_draws += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget,
                    **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    gradient_norm=float(norm), minimum_query_visits=int(sampler.counts.min()))
                trace.append(row); heartbeat(state='observation_cost_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    state = base.read_checkpoint(path)
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete original query coverage required')
    return state


def predict(state, causal, radius):
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Inference refuses targets, validity and endpoint fields')
    radius = np.asarray(radius)
    if radius.shape != causal['envelope'].shape or not np.isfinite(radius).all() or (radius < 0).any():
        raise ValueError('Aligned finite causal candidate/floor radius required')
    pr = state['preprocess']
    model = ObservationCostHead(causal['x'].shape[-1], state['settings']['width'], state['seed'], pr['bias'], state['observation_prior'])
    model.load_state_dict(state['model']); model.eval()
    result = np.empty((len(causal['x']), 7, 5), np.float64); probability = np.empty(len(result))
    pending, size = [], 0
    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        scale = causal['scales'][ix]
        env = (causal['envelope'][ix]/scale[:, None]).astype(np.float32)
        d = (radius[ix]/scale[:, None]).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(z), torch.from_numpy(env), torch.from_numpy(d), torch.from_numpy(seg), len(pending))
        result[ix] = out['moments'].numpy().astype(float)*scale[:, None, None]
        probability[ix] = out['observation_probability'].numpy()
    for group in causal['groups']:
        if pending and size+len(group) > 1024: flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending: flush()
    if not np.isfinite(result).all() or (result < 0).any() or not np.isfinite(probability).all():
        raise FloatingPointError('Finite nonnegative observation-factorized predictions required')
    return dict(costs=result, observation_probability=probability)
