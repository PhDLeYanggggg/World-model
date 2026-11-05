"""Cost-weighted sign learning; scores are not calibrated expected risk."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_kinematic_risk as parent

np, torch, nn, F, base = parent.np, parent.torch, parent.nn, parent.F, parent.base
ARMS = ('continued_mse', 'signed_cost_logistic')


def signed_cost(score, target):
    if score.shape != target.shape or target.requires_grad or not torch.isfinite(target).all():
        raise ValueError('Aligned detached finite signed costs required')
    # At score zero this has the same output gradient (-2*y) as squared error.
    return 4*target.abs()*F.softplus(-target.sign()*score)


def objective(prediction, data, ix, segments, count, arm):
    if arm not in ARMS:
        raise ValueError('Registered MSE versus signed-cost loss required')
    original = parent.parent.objective(prediction, data, ix, segments, count, 'mean')
    if arm == 'continued_mse':
        return original
    legal = torch.from_numpy(data['causal']['allowed'][ix])
    known = torch.from_numpy(data['observed_future'][ix])
    truth = (parent.parent.signed(data['truth_by_arm']['bound_aware'][ix]) /
        torch.from_numpy(parent.parent.normalizer(data['preprocess']))).detach()
    error = torch.where(known[:, None, None], signed_cost(prediction, truth), (prediction-truth).square())
    row = (error.mean(-1)*legal).sum(1)/legal.sum(1).clamp_min(1)
    supported = legal.any(1)
    member = F.one_hot(segments, count).to(prediction.dtype).T
    loss = ((member @ (row*supported))/(member @ supported.float()).clamp_min(1)).mean()
    return dict(total=loss, anchor_MSE=original['total'],
        observed_supervised_rows=original['observed_supervised_rows'],
        unknown_bound_supervised_rows=original['unknown_bound_supervised_rows'])


def fit(data, warm, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or set(data['causal']) != base.CAUSAL_KEYS or warm['arm'] != 'kinematic':
        raise ValueError('Fixed causal schema and common kinematic warm model required')
    if settings['width'] != warm['settings']['width']:
        raise ValueError('Unchanged model capacity required')
    c = data['causal']; z = parent.batch_features(data, 'kinematic')
    model = parent.KinematicRiskHead(settings['width'], warm['seed']); model.load_state_dict(warm['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm='kinematic', objective_arm=arm, settings=settings, identity=identity,
        seed=warm['seed'], preprocess=data['preprocess'], extra_preprocess=data['extra_preprocess'],
        extra_raw_hash=data['extra_raw_hash'], extra_hash=data['extra_hash'],
        fit_ids=c['ids'].copy(), source_sites=sorted(set(c['sites'])), initial_model=copy.deepcopy(model.state_dict()),
        parameters=sum(v.numel() for v in model.parameters()), budget=budget, risk_budget=.02,
        warm_step=warm['step'], optimizer_reused=False, input_hash=parent.digest(z.numpy()),
        target_hash=parent.digest(data['truth_by_arm']['bound_aware'].numpy()),
        observed_mask_hash=parent.digest(data['observed_future']), proposal_hash=parent.digest(c['allowed']),
        temperature=1., logistic_multiplier=4., unknown_bound_objective='unchanged_MSE',
        score_semantics='signed_decision_score_not_calibrated_expected_risk')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed, unknown = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        saved = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(saved[key], value)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); sampler.restore(saved['sampler'])
        torch.set_rng_state(saved['torch_rng'])
        step, elapsed, trace, draws, observed, unknown = (saved[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else stop_at
    if not 0 <= step <= limit <= budget or limit <= 0:
        raise ValueError('Nondecreasing fixed budget required')
    started = time.monotonic()

    def save():
        guard(); base.save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(),
            torch_rng=torch.get_rng_state(), trace=trace, draw_hash=draws,
            observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown,
            threshold_search=False, calibrated_safety_bound=False, expected_harm_calibrated=False))
    if not step:
        save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), arm)
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Nonfinite signed-decision objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, gradient_norm=float(norm),
                    **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='signed_decision_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete original source-query coverage required')
    return base.read_checkpoint(path)


def assert_matched(a, b):
    if (a['objective_arm'], b['objective_arm']) != ARMS:
        raise ValueError('Both registered objective arms required')
    for key in ('settings', 'identity', 'seed', 'preprocess', 'extra_preprocess', 'extra_raw_hash', 'extra_hash',
            'fit_ids', 'source_sites', 'initial_model', 'parameters', 'budget', 'risk_budget', 'warm_step',
            'optimizer_reused', 'input_hash', 'target_hash', 'observed_mask_hash', 'proposal_hash',
            'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])


predict = parent.predict
