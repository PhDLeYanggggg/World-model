"""Same causal cost network; a selected-record excess objective is loss-only."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_amplitude_cost as parent

base = parent.base
ARMS = ('point', 'record_excess')


def trace_losses(terms, extra, total):
    return {**{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
            **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in extra.items()},
            'base_total': float(terms['total'].detach()), 'total': float(total.detach())}


def training_packet(learning, teacher_actions, recordings):
    c = learning['causal']
    choice, records = np.asarray(teacher_actions), np.asarray(recordings)
    if (set(c) != base.base.CAUSAL_KEYS or choice.shape != (len(c['ids']),)
            or choice.dtype.kind not in 'iu' or records.shape != choice.shape
            or np.any((choice < -1) | (choice >= 7))):
        raise ValueError('Exact causal schema and fixed fitting-only teacher decisions required')
    at = np.flatnonzero(choice >= 0)
    if not c['allowed'][at, choice[at]].all():
        raise ValueError('Teacher action must be causally legal')
    names, codes = np.unique(records, return_inverse=True)
    sites = sorted(set(c['sites']))
    record_sources = []
    for name in names:
        present = set(c['sites'][records == name])
        if len(present) != 1:
            raise ValueError('A recording cannot span source groups')
        record_sources.append(sites.index(next(iter(present))))
    return dict(learning, teacher_actions=choice.copy(), record_codes=codes.astype(np.int64),
        record_source_codes=np.asarray(record_sources, np.int64))


def record_excess(prediction, data, ix, *, budget=.02):
    if budget != .02 or prediction.shape != (len(ix), 7, 5):
        raise ValueError('Unchanged2%relative risk and original cost output required')
    choice = torch.from_numpy(data['teacher_actions'][ix])
    active = choice >= 0
    if not active.any():
        return dict(record_excess=prediction.sum()*0, active_records=0,
                    active_sources=0, teacher_selected_rows=0)
    record_ids, inverse = torch.unique(torch.from_numpy(data['record_codes'][ix]), return_inverse=True)
    membership = F.one_hot(inverse, len(record_ids)).to(prediction.dtype).T
    rows = torch.arange(len(ix))
    error = prediction[rows, choice.clamp_min(0)] - data['truth_by_arm']['bound_aware'][ix][rows, choice.clamp_min(0)]
    # Grouping is loss-only. The network still pools within current queries only.
    excess = torch.stack((error[:, 1]-budget*error[:, 2], error[:, 4]-budget*error[:, 3]), -1)
    scale = torch.from_numpy(data['causal']['scales'][ix]).to(prediction.dtype) * active
    denominator = membership @ scale
    rms = torch.from_numpy(data['preprocess']['rms']).to(prediction.dtype)
    units = torch.stack((torch.sqrt(rms[1]**2+(budget*rms[2])**2),
                         torch.sqrt(rms[4]**2+(budget*rms[3])**2)))
    value = ((membership @ (excess*scale[:, None])) / denominator.clamp_min(1e-12)[:, None] / units).square().mean(-1)
    supported = denominator > 0
    sources = torch.from_numpy(data['record_source_codes'])[record_ids]
    terms = [value[(sources == s) & supported].mean() for s in torch.unique(sources[supported])]
    return dict(record_excess=torch.stack(terms).mean(), active_records=int(supported.sum()),
                active_sources=len(terms), teacher_selected_rows=int(active.sum()))


def fit(data, *, loss_arm, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if loss_arm not in ARMS:
        raise ValueError('Fixed point-versus-record objective comparison required')
    c, pr = data['causal'], data['preprocess']
    model = base.base.CostHead(c['x'].shape[-1], settings['width'], seed, pr['bias'], 'context')
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.base.steps_for(c, settings)
    from src.world_model.m3w_harm_refinement import digest
    binding = dict(identity=identity, settings=settings, arm='context', target_arm='bound_aware',
        loss_arm=loss_arm, loss_weight=1., positive_harm_budget=.02,
        seed=seed, preprocess=pr, initial_model=initial, parameters=sum(p.numel() for p in model.parameters()),
        budget=budget, source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy(),
        teacher_sha256=digest(data['teacher_actions']), record_codes_sha256=digest(data['record_codes']),
        targets_sha256=digest(data['truth_by_arm']['bound_aware'].numpy()),
        standardized_features_sha256=digest(data['z'].numpy()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed_draws, bound_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit same-binding resume required')
        old = base.base.read_checkpoint(path)
        for key, value in binding.items():
            base.base.exact(old[key], value)
        model.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler'])
        torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed_draws, bound_draws = (
            old[k] for k in ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('No budget rollback or zero-update result')
    start = time.monotonic()

    def save():
        guard()
        base.base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-start, trace=trace, draw_hash=draws,
            observed_supervised_draws=observed_draws, unknown_bound_supervised_draws=bound_draws,
            unknown_rows_mislabeled_as_observed=0))

    if step == 0:
        save()
    try:
        while step < limit:
            guard()
            ix, seg, qids = sampler.draw()
            segment = torch.from_numpy(seg)
            envelope = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            opt.zero_grad(set_to_none=True)
            prediction = model(data['z'][ix], envelope, segment, len(qids))
            terms = base.objective(prediction, data, ix, segment, len(qids), 'bound_aware')
            extra = record_excess(prediction, data, ix) if loss_arm == 'record_excess' else {}
            total = terms['total'] + extra['record_excess'] if loss_arm == 'record_excess' else terms['total']
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite aggregate cost objective')
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            step += 1
            observed_draws += terms['observed_supervised_rows']
            bound_draws += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget,
                    **trace_losses(terms, extra, total),
                    gradient_norm=float(norm), minimum_query_visits=int(sampler.counts.min()),
                    never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row)
                heartbeat(state='record_excess_cost_training', loss_arm=loss_arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save()
        raise
    result = base.base.read_checkpoint(path)
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Every fitting query needs its complete coverage budget')
    return result


def exact_scientific_state(a, b):
    for key in ('model', 'optimizer', 'sampler', 'draw_hash', 'torch_rng', 'budget', 'step',
                'fit_ids', 'initial_model', 'preprocess', 'settings', 'seed', 'parameters',
                'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.base.exact(a[key], b[key])
