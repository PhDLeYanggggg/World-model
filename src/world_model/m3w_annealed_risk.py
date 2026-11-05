"""Matched risk-head continuation with a fixed, non-selected learning schedule."""
import copy
import hashlib
import math
from pathlib import Path
import time

from src.world_model import m3w_kinematic_risk as parent

np, torch, nn, base = parent.np, parent.torch, parent.nn, parent.base
ARMS = ('constant', 'cosine')


def learning_rate(arm, initial, update, budget, minimum_ratio=.01):
    if arm not in ARMS or not 0 <= update < budget or initial <= 0 or not 0 < minimum_ratio <= 1:
        raise ValueError('Fixed schedule and valid update required')
    if arm == 'constant':
        return initial
    phase = update/max(budget-1, 1)
    return initial*(minimum_ratio+(1-minimum_ratio)*(1+math.cos(math.pi*phase))/2)


def fit(data, warm, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or set(data['causal']) != base.CAUSAL_KEYS or warm['arm'] != 'kinematic':
        raise ValueError('Fixed causal schema and common kinematic warm model required')
    if settings['width'] != warm['settings']['width']:
        raise ValueError('Unchanged capacity required')
    c = data['causal']; z = parent.batch_features(data, 'kinematic')
    model = parent.KinematicRiskHead(settings['width'], warm['seed']); model.load_state_dict(warm['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm='kinematic', schedule_arm=arm, settings=settings, identity=identity,
        seed=warm['seed'], preprocess=data['preprocess'], extra_preprocess=data['extra_preprocess'],
        extra_raw_hash=data['extra_raw_hash'], extra_hash=data['extra_hash'],
        fit_ids=c['ids'].copy(), source_sites=sorted(set(c['sites'])), initial_model=copy.deepcopy(model.state_dict()),
        parameters=sum(v.numel() for v in model.parameters()), budget=budget, risk_budget=.02,
        warm_step=warm['step'], optimizer_reused=False, input_hash=parent.digest(z.numpy()),
        target_hash=parent.digest(data['truth_by_arm']['bound_aware'].numpy()),
        observed_mask_hash=parent.digest(data['observed_future']), proposal_hash=parent.digest(c['allowed']),
        minimum_lr_ratio=.01, objective='unchanged_direct_mean_MSE',
        checkpoint_selection='fixed_last_update_not_selected_on_V')
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
        raise ValueError('Nondecreasing full-budget schedule required')
    started = time.monotonic()

    def save():
        guard(); base.save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(),
            torch_rng=torch.get_rng_state(), trace=trace, draw_hash=draws,
            observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown,
            threshold_search=False, calibrated_safety_bound=False))

    if not step:
        save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
            rate = learning_rate(arm, settings['learning_rate'], step, budget)
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            terms = parent.parent.objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), 'mean')
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Nonfinite risk objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, learning_rate=rate, gradient_norm=float(norm),
                    **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='annealed_risk_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete original source-query coverage required')
    return base.read_checkpoint(path)


def assert_matched(a, b):
    if (a['schedule_arm'], b['schedule_arm']) != ARMS:
        raise ValueError('Both registered schedule arms required')
    for key in ('settings', 'identity', 'seed', 'preprocess', 'extra_preprocess', 'extra_raw_hash', 'extra_hash',
            'fit_ids', 'source_sites', 'initial_model', 'parameters', 'budget', 'risk_budget', 'warm_step',
            'optimizer_reused', 'input_hash', 'target_hash', 'observed_mask_hash', 'proposal_hash',
            'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])


predict = parent.predict
