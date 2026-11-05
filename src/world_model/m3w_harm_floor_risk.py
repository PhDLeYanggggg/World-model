"""Fitting-only harm-weight allocation; inference and scientific rules unchanged."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_joint_decision_risk as previous

parent = previous.parent
np, torch, nn, base = parent.np, parent.torch, parent.nn, parent.base
ARMS = ('joint', 'uniform_harm', 'harm_floor')
predict, full, model_from, attribution = previous.predict, previous.full, previous.model_from, previous.attribution


def positive_weighting(positive, weights):
    x, w = np.asarray(positive, np.float64), np.asarray(weights, np.float64)
    if x.shape != w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Finite aligned own-fitting magnitudes and weights required')
    take = (x > 0) & (w > 0); x, w = x[take], w[take]
    if not len(x):
        raise ValueError('Observed positive fitting support required; no invented floor')
    order = np.argsort(x, kind='stable'); x, w = x[order], w[order]
    floor = float(x[np.searchsorted(np.cumsum(w), w.sum()/2)])
    old_mass = float(np.dot(x, w)); new_mass = float(np.dot(np.maximum(x, floor), w))
    return dict(positive_weighted_median=floor, original_positive_mass=old_mass,
        floor_positive_mass=new_mass, uniform_multiplier=new_mass/old_mass,
        positive_rows=len(x), fitting_only=True, inference_use=False)


def weighting(data, origin, settings):
    c = data['causal']
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], origin['seed'])
    for _ in range(base.steps_for(c, settings)):
        sampler.draw()
    w = parent.query_weights(c, sampler.counts)
    target = (parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        torch.from_numpy(parent.kin.parent.normalizer(data['preprocess']))).numpy()
    p = np.maximum(target, 0).mean(-1)
    selected = c['allowed'] & data['observed_future'][:, None]
    result = positive_weighting(p, w[:, None]*selected)
    return dict(result, planned_query_counts=sampler.counts.copy())


def cost(risk, decision, target, weights, arm):
    if arm not in ARMS:
        raise ValueError('Fixed harm allocation contrast required')
    if arm == 'joint':
        return previous.joint_cost(risk, decision, target)
    p = target.clamp_min(0).mean(-1)
    n = (-target).clamp_min(0).mean(-1)
    y = torch.where(p > 0, p, -n).detach()
    magnitude = y.abs()
    positive = p.clamp_min(weights['positive_weighted_median']) if arm == 'harm_floor' else p*weights['uniform_multiplier']
    magnitude = torch.where(p > 0, positive, magnitude).detach()
    score = torch.maximum(risk, decision).amax(-1)
    return (4*magnitude*parent.F.softplus(-torch.sign(y)*score)).unsqueeze(-1)


def objective(predictions, data, ix, segments, count, arm, weights):
    if arm == 'joint':
        return previous.objective(predictions, data, ix, segments, count, 'joint')
    original = parent.objective(predictions, data, ix, segments, count)
    target = (parent.kin.parent.signed(data['truth_by_arm']['bound_aware'][ix]) /
        torch.from_numpy(parent.kin.parent.normalizer(data['preprocess']))).detach()
    known = torch.from_numpy(data['observed_future'][ix])
    error = torch.where(known[:, None, None], cost(*predictions, target, weights, arm), 0.)
    added = parent.aggregate(error, torch.from_numpy(data['causal']['allowed'][ix]), segments, count)
    return dict(original, total=original['total']+.5*added, composed_observed_cost=added)


def fit(data, origin, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or settings['width'] != origin['settings']['width']:
        raise ValueError('Same separate64 capacity and fixed contrast required')
    z = parent.validate(data, origin); c = data['causal']; weights = weighting(data, origin, settings)
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
        composition_coefficient=.5, threshold_search=False, calibrated_safety_bound=False,
        decision_semantics='joint_sign_cost_not_probability', fitting_harm_weighting=weights)
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
        raise ValueError('Fixed nondecreasing update budget required')
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
        terms = objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), arm, weights)
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite harm-floor objective required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='harm_floor_training', loss_arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget:
        base.exact(sampler.counts, weights['planned_query_counts'])
        if sampler.counts.min() < settings['largest_source_cycles']:
            raise ValueError('Complete planned fitting coverage required')
    return base.read_checkpoint(path)


def assert_matched(states):
    if tuple(s['loss_arm'] for s in states) != ARMS:
        raise ValueError('All three fixed loss arms required')
    for state in states[1:]:
        for key in ('settings', 'identity', 'seed', 'initial_model', 'parameters', 'fit_ids', 'source_sites',
                'preprocess', 'extra_preprocess', 'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash',
                'risk_budget', 'step', 'sampler', 'draw_hash', 'observed_supervised_draws',
                'unknown_bound_supervised_draws', 'fitting_harm_weighting'):
            base.exact(states[0][key], state[key])
