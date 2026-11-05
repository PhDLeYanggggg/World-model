"""Full-fitting optimization diagnostic, not a calibrated risk guarantee."""
import copy
from pathlib import Path
import time

from src.world_model import m3w_kinematic_risk as parent
from src.evaluation.m3w_all_easy_gradient import batches, task_losses
from src.world_model.m3w_frozen_readout_refit import query_weights

np, torch, base = parent.np, parent.torch, parent.base


def prepare(data, origin):
    c = data['causal']; z = parent.batch_features(data, 'kinematic')
    if set(c) != base.CAUSAL_KEYS or origin['arm'] != 'kinematic':
        raise ValueError('Unchanged causal schema and kinematic risk model required')
    for key, value in (('fit_ids', c['ids']), ('input_hash', parent.digest(z.numpy())),
            ('target_hash', parent.digest(data['truth_by_arm']['bound_aware'].numpy())),
            ('proposal_hash', parent.digest(c['allowed'])), ('preprocess', data['preprocess']),
            ('extra_preprocess', data['extra_preprocess'])):
        base.exact(origin[key], value)
    y = parent.parent.signed(data['truth_by_arm']['bound_aware'])/torch.from_numpy(parent.parent.normalizer(data['preprocess']))
    return dict(z=z, target=y.detach().double(), allowed=torch.from_numpy(c['allowed']),
        weights=torch.from_numpy(query_weights(c, origin['sampler']['counts'])),
        batches=list(batches(c)), known=data['observed_future'])


def model_from(state):
    model = parent.KinematicRiskHead(state['settings']['width'], state['seed']).double()
    model.load_state_dict(state['model']); model.eval()
    return model


def full(model, prepared, gradient=False, guard=lambda: None):
    if gradient:
        model.zero_grad(set_to_none=True)
    total, regret = np.zeros(2), np.zeros(2)
    false_safe = np.zeros(2, dtype=np.int64)
    for ix, seg in prepared['batches']:
        guard()
        with torch.set_grad_enabled(gradient):
            pred = model(prepared['z'][ix].double(), torch.from_numpy(seg), int(seg.max())+1)
            terms = task_losses(pred, prepared['target'][ix], prepared['allowed'][ix], prepared['weights'][ix])
            if not torch.isfinite(terms).all():
                raise FloatingPointError('Nonfinite complete fitting loss')
            if gradient:
                terms.mean().backward()
        total += terms.detach().numpy()
        p, y = pred.detach().numpy(), prepared['target'][ix].numpy()
        known = prepared['known'][ix, None, None] & prepared['allowed'][ix].numpy()[..., None]
        wrong = ((p <= 0) & (y > 0)) | ((p > 0) & (y < 0))
        regret += (prepared['weights'][ix].numpy()[:, None, None]*abs(y)*wrong*known).sum((0, 1))
        false_safe += (known & (p <= 0) & (y > 0)).sum((0, 1))
    out = dict(mean_MSE=float(total.mean()), task_MSE=total.tolist(),
        observed_sign_regret=regret.tolist(), false_safe_channels=false_safe.tolist())
    if gradient:
        grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        if not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite complete fitting gradient')
        out.update(gradient_inf=float(grad.abs().max()), gradient_norm=float(grad.norm()))
    return out


def fit(data, origin, *, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    prepared = prepare(data, origin); model = model_from(origin)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, max_eval=25,
        tolerance_grad=settings['tolerance_grad'], tolerance_change=settings['tolerance_change'],
        history_size=settings['history_size'], line_search_fn='strong_wolfe')
    binding = dict(arm='kinematic', solver='full_B_LBFGS', precision='float64',
        settings=dict(origin['settings']), solver_settings=settings, identity=identity,
        seed=origin['seed'], preprocess=origin['preprocess'], extra_preprocess=origin['extra_preprocess'],
        fit_ids=origin['fit_ids'], source_sites=origin['source_sites'],
        input_hash=origin['input_hash'], target_hash=origin['target_hash'], proposal_hash=origin['proposal_hash'],
        initial_model=copy.deepcopy(model.state_dict()), risk_budget=.02,
        query_weights_hash=parent.digest(prepared['weights'].numpy()),
        query_counts=origin['sampler']['counts'], parameters=sum(p.numel() for p in model.parameters()),
        objective='unchanged_empirical_query_weighted_mean_MSE',
        optimizer_state_reused=False, weight_decay=0., gradient_clipping=False,
        checkpoint_selection='fixed_fit_budget_or_fitting_stationarity_not_V')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, changed_steps, passes, elapsed, trace, reason = 0, 0, 0, 0., [], 'in_progress'
    start = time.monotonic()
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        saved = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(saved[key], value)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        step, changed_steps, passes, elapsed, trace, reason = (saved[k] for k in
            ('step', 'changed_parameter_steps', 'full_gradient_passes', 'seconds', 'trace', 'stop_reason'))
        initial, latest = saved['initial_fitting'], saved['final_fitting']
    else:
        initial = full(model, prepared, True, guard); passes += 1; latest = initial
    limit = settings['outer_steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['outer_steps'] or limit <= 0:
        raise ValueError('Nondecreasing registered budget required')

    def save():
        guard()
        opt = optimizer.state_dict()
        counters = next(iter(opt['state'].values()), {})
        base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt,
            torch_rng=torch.get_rng_state(), step=step, changed_parameter_steps=changed_steps,
            full_gradient_passes=passes, lbfgs_iterations=int(counters.get('n_iter', 0)),
            lbfgs_function_evaluations=int(counters.get('func_evals', 0)),
            seconds=elapsed+time.monotonic()-start, trace=trace, stop_reason=reason,
            initial_fitting=initial, final_fitting=latest,
            full_fitting_row_gradient_occurrences=passes*len(prepared['z']),
            threshold_search=False, calibrated_safety_bound=False))

    def closure():
        nonlocal passes
        loss = full(model, prepared, True, guard); passes += 1
        return torch.tensor(loss['mean_MSE'], dtype=torch.float64)

    if not path.exists():
        save()
    # A line-search interruption must not serialize transient trial parameters.
    # Only completed outer steps reach save(); an interrupted run resumes the
    # most recent atomic checkpoint, including its quasi-Newton history.
    while step < limit and reason != 'fitting_stationarity':
        guard(); before = [p.detach().clone() for p in model.parameters()]
        previous = latest['mean_MSE']; optimizer.step(closure)
        latest = full(model, prepared, True, guard); passes += 1
        if latest['mean_MSE'] > previous+1e-10*max(1., abs(previous)):
            raise FloatingPointError('Complete objective increased after accepted line search; checkpoint preserved')
        changed = any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
        changed_steps += int(changed); step += 1
        reason = 'fitting_stationarity' if latest['gradient_inf'] <= settings['tolerance_grad'] else 'in_progress'
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['outer_steps'] or reason == 'fitting_stationarity':
            row = dict(step=step, full_gradient_passes=passes, changed_parameter_steps=changed_steps, **latest)
            trace.append(row); heartbeat(state='full_objective_risk_training', **row)
        if step == settings['outer_steps'] and reason == 'in_progress':
            reason = 'budget_exhausted_not_proven_converged'
        if step % settings['checkpoint_every'] == 0 or step == limit or reason == 'fitting_stationarity':
            save()
    return base.read_checkpoint(path)


def predict(state, causal, *, initial=False):
    if set(causal) != base.CAUSAL_KEYS or causal['x'].shape[-1] != 428 or state['precision'] != 'float64':
        raise ValueError('Only frozen float64 model and causal schema may enter inference')
    model = model_from(dict(state, model=state['initial_model']) if initial else state)
    pr, ep = state['preprocess'], state['extra_preprocess']
    result = np.empty((len(causal['x']), 7, 2), np.float64)
    for ix, seg in batches(causal):
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0
        extra = np.clip((parent.motion_basis(causal['x'][ix])-ep['mean'])/ep['std'], -ep['clip'], ep['clip']).astype(np.float32)
        with torch.no_grad():
            value = model(torch.from_numpy(np.concatenate((z, extra), -1)).double(),
                torch.from_numpy(seg), int(seg.max())+1).numpy()
        result[ix] = value*parent.parent.normalizer(pr)[None, None, :]*causal['scales'][ix, None, None]
    if not np.isfinite(result).all():
        raise FloatingPointError('Finite causal risk scores required')
    return result


def fitting_screen(rows):
    decisions = []
    for r in rows:
        a, b = r['initial_fitting'], r['final_fitting']
        improvement = (a['mean_MSE']-b['mean_MSE'])/max(a['mean_MSE'], 1e-15)
        no_worse_sign_cost = all(x <= y+1e-12 for x, y in zip(b['observed_sign_regret'], a['observed_sign_regret']))
        decisions.append(dict(job=r['job'], mean_MSE_improvement_fraction=improvement,
            all_and_easy_sign_cost_not_worse=no_worse_sign_cost,
            passed=improvement >= .01 and no_worse_sign_cost))
    return dict(contexts=decisions, followup_readout_allowed=len(rows) == 3 and all(r['passed'] for r in decisions),
        safety_certified=False, original_rules_unchanged=True)
