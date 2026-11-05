"""Fixed warm-error replay weighting; not calibrated expected harm or a guarantee."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_kinematic_risk as parent

np, torch, nn, F, base = parent.np, parent.torch, parent.nn, parent.F, parent.base
ARMS = ('continued_mse', 'false_safe_priority')
MULTIPLIER = 10.


def prepare(data, warm):
    if warm['arm'] != 'kinematic':
        raise ValueError('Same frozen kinematic warm model in both objective arms')
    c = data['causal']
    for key, value in (('fit_ids', c['ids']), ('preprocess', data['preprocess']),
            ('extra_preprocess', data['extra_preprocess']), ('target_hash', parent.digest(data['truth_by_arm']['bound_aware'].numpy())),
            ('extra_hash', data['extra_hash']), ('common_input_hash', parent.digest(data['z'].numpy())),
            ('proposal_hash', parent.digest(c['allowed']))):
        base.exact(warm[key], value)
    predicted = parent.predict(warm, c)
    truth = parent.parent.signed(data['truth_by_arm']['bound_aware']).numpy()
    mask = ((predicted <= 0) & (truth > 0) & data['observed_future'][:, None, None]
        & c['allowed'][..., None])
    return dict(data, priority_mask=torch.from_numpy(mask), priority_mask_hash=parent.digest(mask),
        warm_prediction_hash=parent.digest(predicted))


def objective(prediction, data, ix, segments, count, arm):
    if arm not in ARMS:
        raise ValueError('Registered objective pair required')
    anchor = parent.parent.objective(prediction, data, ix, segments, count, 'mean')
    legal = torch.from_numpy(data['causal']['allowed'][ix])
    truth = parent.parent.signed(data['truth_by_arm']['bound_aware'][ix])/torch.from_numpy(parent.parent.normalizer(data['preprocess']))
    priority = data['priority_mask'][ix]
    if priority.dtype != torch.bool or priority.shape != prediction.shape:
        raise ValueError('Frozen fitting-only two-channel priority mask required')
    if (priority & ~torch.from_numpy(data['observed_future'][ix])[:, None, None]).any():
        raise ValueError('Unknown outcomes retain anchor bounds, never observed-error priority')
    row = (((prediction-truth.detach()).square()*priority).mean(-1)*legal).sum(1)/legal.sum(1).clamp_min(1)
    supported = legal.any(1)
    member = F.one_hot(segments, count).to(prediction.dtype).T
    extra = ((member @ row)/(member @ supported.float()).clamp_min(1)).mean()
    total = anchor['total'] if arm == 'continued_mse' else anchor['total']+(MULTIPLIER-1)*extra
    return dict(anchor, total=total, anchor=anchor['total'], priority_MSE=extra,
        priority_channels=int((priority & legal[..., None]).sum()))


def initialize(warm):
    model = parent.KinematicRiskHead(warm['settings']['width'], warm['seed'])
    model.load_state_dict(warm['model'])
    return model


def fit(data, warm, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or settings['width'] != warm['settings']['width']:
        raise ValueError('Fixed objective and unchanged architecture required')
    z = parent.batch_features(data, 'kinematic'); c = data['causal']; model = initialize(warm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm='kinematic', objective_arm=arm, priority_multiplier=MULTIPLIER,
        settings=settings, identity=identity, seed=warm['seed'], preprocess=data['preprocess'],
        extra_preprocess=data['extra_preprocess'], extra_raw_hash=data['extra_raw_hash'], extra_hash=data['extra_hash'],
        fit_ids=c['ids'].copy(), source_sites=sorted(set(c['sites'])), initial_model=copy.deepcopy(model.state_dict()),
        parameters=sum(v.numel() for v in model.parameters()), budget=budget, risk_budget=.02,
        warm_step=warm['step'], optimizer_reused=False, input_hash=parent.digest(z.numpy()),
        target_hash=parent.digest(data['truth_by_arm']['bound_aware'].numpy()), proposal_hash=parent.digest(c['allowed']),
        priority_mask_hash=data['priority_mask_hash'], warm_prediction_hash=data['warm_prediction_hash'])
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
                raise FloatingPointError('Nonfinite fixed-priority loss')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, gradient_norm=float(norm),
                    **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='fixed_priority_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete original source-query coverage required')
    return base.read_checkpoint(path)


def assert_matched(a, b):
    if (a['objective_arm'], b['objective_arm']) != ARMS:
        raise ValueError('Both objective arms required')
    for key in ('settings', 'identity', 'seed', 'preprocess', 'extra_preprocess', 'extra_raw_hash', 'extra_hash',
            'fit_ids', 'source_sites', 'initial_model', 'parameters', 'budget', 'risk_budget', 'warm_step',
            'optimizer_reused', 'input_hash', 'target_hash', 'proposal_hash', 'priority_mask_hash',
            'warm_prediction_hash', 'step', 'sampler', 'draw_hash', 'observed_supervised_draws',
            'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])


predict = parent.predict
