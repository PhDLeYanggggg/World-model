"""Within-query useful/harmful contrast; original decisions and rules unchanged."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_joint_decision_risk as previous

parent = previous.parent
np, torch, nn, base = parent.np, parent.torch, parent.nn, parent.base
ARMS = ('joint', 'paired_point', 'paired_rank')
predict, full, model_from, attribution = previous.predict, previous.full, previous.model_from, previous.attribution


def pairs(target, allowed, observed, segments, count, step):
    """Deterministic cyclic pairs stay inside each complete causal query."""
    positive = target.clamp_min(0).mean(-1)
    negative = (-target).clamp_min(0).mean(-1)
    y = torch.where(positive > 0, positive, -negative).detach()
    if allowed.sum(1).max().item() > 1 or step < 0:
        raise ValueError('One frozen proposal per row and nonnegative update required')
    valid = allowed & observed[:, None]
    result = []
    for group in range(count):
        within = valid & (segments == group)[:, None]
        p = torch.nonzero(within & (y > 0), as_tuple=False)
        n = torch.nonzero(within & (y < 0), as_tuple=False)
        if not len(p) or not len(n):
            continue
        k = torch.arange(max(len(p), len(n)))
        pi, ni = p[k % len(p)], n[(k+step) % len(n)]
        w = (y[pi[:, 0], pi[:, 1]]-y[ni[:, 0], ni[:, 1]])/2
        result.append((group, pi, ni, w))
    return result


def contrast(score_positive, score_negative, weight, arm):
    if arm == 'paired_point':
        value = (parent.F.softplus(-score_positive)+parent.F.softplus(score_negative))/2
    elif arm == 'paired_rank':
        value = parent.F.softplus((score_negative-score_positive)/2)
    else:
        raise ValueError('Fixed paired objective required')
    return 4*weight.detach()*value


def objective(predictions, data, ix, segments, count, arm, step):
    if arm not in ARMS:
        raise ValueError('Fixed three-arm contrast required')
    original = previous.objective(predictions, data, ix, segments, count, 'joint')
    if arm == 'joint':
        return original
    y = (parent.kin.parent.signed(data['truth_by_arm']['bound_aware'][ix]) /
        torch.from_numpy(parent.kin.parent.normalizer(data['preprocess']))).detach()
    legal, known = torch.from_numpy(data['causal']['allowed'][ix]), torch.from_numpy(data['observed_future'][ix])
    group_pairs = pairs(y, legal, known, segments, count, step)
    score = torch.maximum(*predictions).amax(-1)
    added = score.sum()*0
    pair_count = 0
    for _, p, n, w in group_pairs:
        added = added+contrast(score[p[:, 0], p[:, 1]], score[n[:, 0], n[:, 1]], w, arm).mean()/count
        pair_count += len(p)
    return dict(original, total=original['total']+.5*added,
        paired_cost=added, paired_queries=len(group_pairs), paired_rows=pair_count)


def fit(data, origin, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or settings['width'] != origin['settings']['width']:
        raise ValueError('Same separate64 capacity and fixed arms required')
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
        paired_coefficient=.5, pairing='within_query_cyclic_step_offset', threshold_search=False,
        calibrated_safety_bound=False, decision_semantics='joint_sign_cost_not_probability')
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
        terms = objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), arm, step)
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite paired decision objective required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='paired_decision_training', loss_arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete planned query coverage required')
    return base.read_checkpoint(path)


def assert_matched(states):
    if tuple(s['loss_arm'] for s in states) != ARMS:
        raise ValueError('All fixed arms required')
    for state in states[1:]:
        for key in ('settings', 'identity', 'seed', 'initial_model', 'parameters', 'fit_ids', 'source_sites',
                'preprocess', 'extra_preprocess', 'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash',
                'risk_budget', 'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
            base.exact(states[0][key], state[key])
