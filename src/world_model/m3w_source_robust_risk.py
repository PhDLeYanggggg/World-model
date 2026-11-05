"""Matched neural continuation with a fixed training-source worst-loss mixture."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_rollout_history_risk as parent

np, torch, nn, F, base = parent.np, parent.torch, parent.nn, parent.F, parent.base
ARMS = ('mean_continue', 'source_robust')
COEFFICIENT = .25
predict, model_from, full, attribution = parent.predict, parent.model_from, parent.full, parent.attribution


def query_terms(predictions, data, ix, segments, count):
    risk, decision = predictions
    allowed = torch.from_numpy(data['causal']['allowed'][ix])
    known = torch.from_numpy(data['observed_future'][ix])
    target = (parent.parent.kin.parent.signed(data['truth_by_arm']['bound_aware'][ix]) /
        torch.from_numpy(parent.parent.kin.parent.normalizer(data['preprocess']))).detach()
    membership = F.one_hot(segments, count).to(risk.dtype).T
    supported = allowed.any(1)
    size = (membership @ supported.float()).clamp_min(1)
    def reduce(error):
        row = (error.mean(-1)*allowed).sum(1)/allowed.sum(1).clamp_min(1)
        return (membership @ (row*supported))/size
    r = reduce((risk-target).square())
    d = reduce(torch.where(known[:, None, None], parent.parent.signed_cost(decision, target),
        (decision-target).square()))
    j = reduce(torch.where(known[:, None, None], parent.previous.previous.joint_cost(risk, decision, target), 0.))
    return dict(risk_MSE=r, decision_objective=d, composed_observed_cost=j, total=(r+d+j)/2)


def objective(predictions, data, ix, segments, count, arm, sources):
    if arm not in ARMS or np.shape(sources) != (count,):
        raise ValueError('Fixed source loss arm and query-aligned source names required')
    original = parent.previous.previous.objective(predictions, data, ix, segments, count, 'joint')
    q = query_terms(predictions, data, ix, segments, count)['total']
    losses = torch.stack([q[torch.from_numpy(sources == site)].mean() for site in sorted(set(sources))])
    worst = losses.max()
    total = original['total'] if arm == 'mean_continue' else (1-COEFFICIENT)*original['total']+COEFFICIENT*worst
    return dict(original, original_total=original['total'], total=total, worst_source_objective=worst,
        sources_in_batch=len(losses))


def validate(data, warm):
    parent.parent.validate(data, warm)
    if warm['feature_arm'] != 'relational' or warm['risk_budget'] != .02:
        raise ValueError('Verified relational neural model and unchanged risk budget required')
    for key in ('relation_hash', 'relation_raw_hash', 'relation_preprocess'):
        base.exact(warm[key], data[key])
    c = data['causal']; sources = np.asarray([c['sites'][ids[0]] for ids in c['groups']])
    if any(len(set(c['sites'][ids])) != 1 for ids in c['groups']) or len(set(sources)) < 2:
        raise ValueError('Homogeneous queries and at least two fitting sources required')
    return parent.parent.kin.batch_features(parent.extended(data, 'relational'), 'kinematic'), sources


def fit(data, warm, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or settings['width'] != warm['settings']['width']:
        raise ValueError('Matched continuation of the same architecture required')
    z, query_sources = validate(data, warm); c = data['causal']; model = model_from(warm)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = base.steps_for(c, settings)
    fields = ('seed', 'preprocess', 'extra_preprocess', 'relation_preprocess', 'relation_hash', 'relation_raw_hash',
        'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash', 'fit_ids', 'source_sites', 'parameters')
    binding = dict(**{k:warm[k] for k in fields}, arm='kinematic', feature_arm='relational',
        source_loss_arm=arm, settings=settings, identity=identity, initial_model=copy.deepcopy(model.state_dict()),
        parent_step=warm['step'], parent_model_hash=hashlib.sha256(b''.join(
            v.detach().numpy().tobytes() for v in warm['model'].values())).hexdigest(),
        optimizer_reused=False, query_sources=query_sources, risk_budget=.02, budget=budget,
        coefficient=COEFFICIENT, checkpoint_selection='fixed_last_iterate_not_V',
        original_bounds_preserved=True, threshold_search=False, calibrated_safety_bound=False,
        decision_semantics='unchanged_joint_sign_not_probability')
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
        raise ValueError('Nondecreasing fixed continuation budget required')
    started = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step, seconds=elapsed+time.monotonic()-started,
            trace=trace, draw_hash=draws, observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown))
    if not step:
        save()
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
        optimizer.zero_grad(set_to_none=True)
        terms = objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), arm, query_sources[qids])
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite source-robust loss required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k,v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='neural_source_risk_training', source_loss_arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('All fitting queries must receive registered coverage')
    return base.read_checkpoint(path)


def source_metrics(state, data, guard=lambda: None, initial=False):
    z, sites = validate(data, state); model = model_from(state, initial)
    c = data['causal']; rows = []; r, d = (np.empty((len(c['ids']), 7, 2), np.float32) for _ in range(2))
    for ix, seg in parent.parent.batches(c):
        guard()
        with torch.no_grad():
            prediction = model(z[ix], torch.from_numpy(seg), int(seg.max())+1)
            terms = query_terms(prediction, data, ix, torch.from_numpy(seg), int(seg.max())+1)
        rows.append({k:v.numpy() for k,v in terms.items()}); r[ix], d[ix] = (v.numpy() for v in prediction)
    values = {k:np.concatenate([row[k] for row in rows]).astype(float) for k in rows[0]}
    counts = state['sampler']['counts'].astype(float)
    if len(counts) != len(sites):
        raise ValueError('Complete fixed query evaluation required')
    weight = parent.parent.query_weights(c, counts)
    target = (parent.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        torch.from_numpy(parent.parent.kin.parent.normalizer(data['preprocess']))).numpy()
    by_source = {}
    for site in sorted(set(sites)):
        at = sites == site; weights = counts[at]/counts[at].sum()
        row = {k:float(v[at]@weights) for k,v in values.items()}
        mass = counts[at].sum()/counts.sum()
        row['joint'] = parent.previous.previous.attribute(r, d, target,
            c['allowed'] & (c['sites'] == site)[:, None], data['observed_future'], weight/mass)
        row['queries'] = int(at.sum()); by_source[str(site)] = row
    return dict(by_source=by_source, worst_source_objective=max(v['total'] for v in by_source.values()),
        weighted_original_objective=float(values['total']@counts/counts.sum()),
        worst_source_joint_cost=max(v['joint']['total_joint_cost'] for v in by_source.values()))


def assert_matched(states):
    if tuple(s['source_loss_arm'] for s in states) != ARMS:
        raise ValueError('Both prespecified source-loss arms required')
    for key in ('settings', 'identity', 'seed', 'initial_model', 'parameters', 'fit_ids', 'source_sites',
        'preprocess', 'extra_preprocess', 'relation_preprocess', 'relation_hash', 'relation_raw_hash',
        'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash', 'risk_budget', 'step',
        'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws', 'parent_model_hash'):
        base.exact(states[0][key], states[1][key])


def screen(rows, jobs):
    comparisons = []
    if len(rows) != 2*len(jobs) or len({(r['job'], r['arm']) for r in rows}) != len(rows):
        raise ValueError('Every matched fit required')
    for job in jobs:
        a, b = (next(r for r in rows if r['job'] == job['key'] and r['arm'] == arm) for arm in ARMS)
        old, new = a['final_sources']['worst_source_objective'], b['final_sources']['worst_source_objective']
        gain = 1-new/old if old > 0 else None
        cost = b['joint_attribution']['total_joint_cost'] <= a['joint_attribution']['total_joint_cost']
        harm = b['joint_attribution']['false_safe_cost'] <= a['joint_attribution']['false_safe_cost']
        comparisons.append(dict(job=job['key'], worst_source_objective_reduction=gain,
            joint_cost_nonworse=bool(cost), false_safe_cost_nonworse=bool(harm),
            passed=bool(gain is not None and gain >= .01 and cost and harm)))
    return dict(contexts=comparisons, followup_readout_allowed=sum(r['passed'] for r in comparisons) >= 2,
        scientific_safety_pass=False, threshold_search=False)
