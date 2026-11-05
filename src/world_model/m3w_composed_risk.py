"""Train the per-channel conjunction without changing risk or support thresholds."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_decoupled_risk as parent

np, torch, nn, base = parent.np, parent.torch, parent.nn, parent.base
ARMS = ('independent', 'composed')
predict, full, model_from = parent.predict, parent.full, parent.model_from


def composed_cost(risk, decision, target):
    return parent.signed_cost(torch.maximum(risk, decision), target)


def objective(predictions, data, ix, segments, count, arm):
    if arm not in ARMS:
        raise ValueError('Prespecified loss contrast required')
    original = parent.objective(predictions, data, ix, segments, count)
    added = original['total'].new_zeros(())
    if arm == 'composed':
        risk, decision = predictions
        target = (parent.kin.parent.signed(data['truth_by_arm']['bound_aware'][ix]) /
            torch.from_numpy(parent.kin.parent.normalizer(data['preprocess']))).detach()
        known = torch.from_numpy(data['observed_future'][ix])
        cost = torch.where(known[:, None, None], composed_cost(risk, decision, target), 0.)
        added = parent.aggregate(cost, torch.from_numpy(data['causal']['allowed'][ix]), segments, count)
    return dict(original, total=original['total']+.5*added, composed_observed_cost=added)


def fit(data, origin, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or settings['width'] != origin['settings']['width']:
        raise ValueError('Unchanged separate64 capacity and registered loss required')
    z = parent.validate(data, origin); c = data['causal']
    model = parent.initialize(origin, 'separate64')
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], origin['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm='kinematic', variant='separate64', loss_arm=arm, settings=settings, identity=identity,
        seed=origin['seed'], preprocess=data['preprocess'], extra_preprocess=data['extra_preprocess'],
        fit_ids=c['ids'], source_sites=sorted(set(c['sites'])), initial_model=copy.deepcopy(model.state_dict()),
        input_hash=origin['input_hash'], target_hash=origin['target_hash'], proposal_hash=origin['proposal_hash'],
        observed_mask_hash=origin['observed_mask_hash'], parameters=sum(p.numel() for p in model.parameters()),
        risk_budget=.02, budget=budget, checkpoint_selection='fixed_last_iterate_not_V',
        original_bounds_preserved=True, unknown_supervision_coefficients_unchanged=True,
        composition_coefficient=.5 if arm == 'composed' else 0., threshold_search=False,
        calibrated_safety_bound=False, decision_semantics='signed_cost_logit_not_probability')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed, unknown = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        old = base.read_checkpoint(path)
        for k, v in binding.items():
            base.exact(old[k], v)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed, unknown = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else stop_at
    if not 0 <= step <= limit <= budget or limit <= 0:
        raise ValueError('Fixed nondecreasing budget required')
    started = time.monotonic()

    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, draw_hash=draws,
            observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown))

    if not step:
        save()
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
        optimizer.zero_grad(set_to_none=True)
        terms = objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), arm)
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite composed objective required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='composed_risk_training', loss_arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete planned fitting coverage required')
    return base.read_checkpoint(path)


def control_equal(a, b):
    for key in ('model', 'optimizer', 'sampler', 'torch_rng', 'draw_hash', 'initial_model', 'parameters',
            'fit_ids', 'source_sites', 'preprocess', 'extra_preprocess', 'input_hash', 'target_hash',
            'proposal_hash', 'observed_mask_hash', 'step', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])
    if len(a['trace']) != len(b['trace']):
        raise AssertionError('Same completed training trace required')
    for left, right in zip(a['trace'], b['trace']):
        for key, value in right.items():
            base.exact(left[key], value)


def assert_matched(states):
    if tuple(s['loss_arm'] for s in states) != ARMS:
        raise ValueError('Both fixed loss arms required')
    a, b = states
    for key in ('settings', 'identity', 'seed', 'initial_model', 'parameters', 'fit_ids', 'source_sites',
            'preprocess', 'extra_preprocess', 'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash',
            'risk_budget', 'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])
