"""Full-fitting weighted least squares for a frozen neural risk representation."""
import copy
import hashlib
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before importing numerical libraries')

import numpy as np
import torch
from src.world_model import m3w_direct_tail_risk as parent

base = parent.base


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def query_weights(causal, counts):
    """Match the empirical mean of the warm run's sampled query objectives."""
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Exact causal schema required')
    counts = np.asarray(counts)
    if (counts.shape != (len(causal['groups']),) or (counts <= 0).any()
            or not np.isfinite(counts).all() or not np.equal(counts, counts.astype(np.int64)).all()):
        raise ValueError('Positive original integer query visits required')
    mask = causal['allowed']
    if mask.dtype != bool or (mask.sum(1) > 1).any():
        raise ValueError('Only one frozen foreign proposal per row')
    supported = mask.any(1)
    w = np.zeros(len(mask), np.float64)
    seen = np.zeros(len(mask), np.int64)
    for g, n in zip(causal['groups'], counts):
        seen[g] += 1
        active = np.asarray(g)[supported[g]]
        if len(active):
            w[active] = n/(counts.sum()*len(active))
    if not np.all(seen == 1):
        raise ValueError('Each causal row must occur in exactly one query')
    return w


def extract(warm, causal, *, heartbeat=lambda **_: None, guard=lambda: None):
    if set(causal) != base.CAUSAL_KEYS or causal['x'].shape[-1] != 428:
        raise ValueError('Inference accepts only the original causal schema')
    if warm['arm'] != 'mean' or warm['risk_budget'] != .02:
        raise ValueError('Only the frozen direct-mean neural representation')
    model = parent.DirectRiskHead(428, warm['settings']['width'], warm['seed'])
    model.load_state_dict(warm['model']); model.eval()
    pr, width = warm['preprocess'], warm['settings']['width']
    mask = causal['allowed']
    if (mask.sum(1) > 1).any():
        raise ValueError('Extract only frozen proposal-aligned features')
    choice = np.where(mask.any(1), mask.argmax(1), -1)
    design = np.zeros((len(choice), width+1), np.float64)
    prediction = np.zeros((len(choice), 2), np.float64)
    captured = []
    hook = model.excess.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    pending, size, processed = [], 0, 0

    def flush():
        nonlocal processed
        guard()
        ix = np.concatenate(pending)
        seg = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0
        captured.clear()
        with torch.no_grad():
            p = model(torch.from_numpy(z), torch.from_numpy(seg), len(pending)).numpy()
        if len(captured) != 1:
            raise ValueError('One unmodified readout invocation required')
        h = captured[0].numpy()
        at = np.flatnonzero(choice[ix] >= 0); action = choice[ix[at]]
        design[ix[at], :width] = h[at, action]
        design[ix[at], width] = 1.
        prediction[ix[at]] = p[at, action]
        processed += len(ix)
        if processed == len(choice) or processed % 10000 < len(ix):
            heartbeat(state='extract_frozen_fitting_representation', processed=processed, total=len(choice))

    try:
        for group in causal['groups']:
            if pending and size+len(group) > 1024:
                flush(); pending, size = [], 0
            pending.append(group); size += len(group)
        if pending:
            flush()
    finally:
        hook.remove()
    if not np.isfinite(design).all() or not np.isfinite(prediction).all():
        raise FloatingPointError('Finite causal design and frozen outputs required')
    return dict(design=design, prediction=prediction, choice=choice)


def solve(design, target, weights, old_coefficients, *, rcond=1e-10):
    x, y, w, old = map(lambda a: np.asarray(a, dtype=np.float64), (design, target, weights, old_coefficients))
    if (x.ndim != 2 or y.shape != (len(x), 2) or w.shape != (len(x),)
            or old.shape != (x.shape[1], 2) or any(not np.isfinite(v).all() for v in (x, y, w, old))
            or (w < 0).any() or w.sum() <= 0 or rcond != 1e-10):
        raise ValueError('Finite aligned fixed fitting problem required')
    active = w > 0
    root = np.sqrt(w[active])[:, None]
    a, b = root*x[active], root*y[active]
    coefficients, _, rank, values = np.linalg.lstsq(a, b, rcond=rcond)
    mse = lambda beta: float(np.sum(w[:, None]*(x@beta-y)**2)/2)
    before, after = mse(old), mse(coefficients)
    if after > before+1e-8*max(1., before):
        raise FloatingPointError('Numerical refit did not minimize the frozen data objective')
    gradient = a.T@(a@coefficients-b)
    stats = dict(algorithm='float64_SVD_weighted_least_squares', rcond=rcond,
        active_rows=int(active.sum()), columns=x.shape[1], retained_rank=int(rank),
        singular_values=values.tolist(), condition_retained=float(values[0]/values[rank-1]) if rank else None,
        empirical_weight_mass=float(w.sum()), old_linear_float64_MSE=before,
        solved_float64_MSE=after, normal_equation_max_abs=float(np.max(np.abs(gradient))),
        relative_normal_residual=float(np.linalg.norm(gradient)/(1+np.linalg.norm(a.T@b))),
        coefficient_max_abs=float(np.max(np.abs(coefficients))))
    return coefficients, stats


def refit(data, warm, *, identity, path, heartbeat=lambda **_: None, guard=lambda: None, resume=False):
    c, pr = data['causal'], data['preprocess']
    if warm['arm'] != 'mean' or warm['step'] != 2000 or warm['budget'] != 2000:
        raise ValueError('Completed registered warm mean head required')
    base.exact(warm['preprocess'], pr); base.exact(warm['fit_ids'], c['ids'])
    for key, value in dict(input_hash=digest(data['z'].numpy()),
            target_hash=digest(data['truth_by_arm']['bound_aware'].numpy()), proposal_mask_hash=digest(c['allowed'])).items():
        base.exact(warm[key], value)
    counts = warm['sampler']['counts']
    if int(counts.sum()) != warm['step']*warm['settings']['query_batch_size']:
        raise ValueError('Original full update/query draw counts required')
    binding = dict(identity=identity, algorithm='frozen_neural_readout_svd_v1',
        input_hash=warm['input_hash'], target_hash=warm['target_hash'], proposal_mask_hash=warm['proposal_mask_hash'],
        query_counts_hash=digest(counts), learned_parameters=2*(warm['settings']['width']+1),
        new_encoder_updates=0, new_adam_updates=0, original_rules_unchanged=True,
        source_sites=warm['source_sites'], fit_ids=warm['fit_ids'], preprocess=pr,
        settings=warm['settings'], seed=warm['seed'], arm='mean', risk_budget=.02)
    path = Path(path)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required for an existing completed fit')
        saved = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(saved[key], value)
        return saved
    started = time.monotonic(); guard()
    features = extract(warm, c, heartbeat=heartbeat, guard=guard)
    w = query_weights(c, counts); choice = features['choice']; rows = np.flatnonzero(choice >= 0)
    y = np.zeros((len(choice), 2), np.float64)
    # Proposal alignment and hidden inputs are fixed before outcomes enter the solve.
    truth = parent.signed(data['truth_by_arm']['bound_aware'])/torch.from_numpy(parent.normalizer(pr))
    y[rows] = truth[rows, choice[rows]].numpy()
    old = np.vstack((warm['model']['excess.weight'].numpy().T, warm['model']['excess.bias'].numpy()))
    heartbeat(state='solving_readout', rows=len(rows), columns=old.shape[0])
    coefficients, diagnostics = solve(features['design'], y, w, old)
    model = copy.deepcopy(warm['model'])
    model['excess.weight'] = torch.from_numpy(coefficients[:-1].T.astype(np.float32).copy())
    model['excess.bias'] = torch.from_numpy(coefficients[-1].astype(np.float32).copy())
    for key in model:
        if not key.startswith('excess.') and not torch.equal(model[key], warm['model'][key]):
            raise ValueError('Encoder parameters must remain bitwise frozen')
    state = dict(**binding, model=model, warm_model=warm['model'], parameter_count=warm['parameters'])
    # Judge the actual saved float32 inference path, not only a float64 solver surrogate.
    after = extract(state, c, heartbeat=heartbeat, guard=guard)['prediction']
    mse = lambda p: float(np.sum(w[:, None]*(p-y)**2)/2)
    before_loss, after_loss = mse(features['prediction']), mse(after)
    if after_loss > before_loss+1e-5*max(1., before_loss):
        raise FloatingPointError('Saved inference does not improve the frozen fitting objective')
    diagnostics.update(original_actual_float32_MSE=before_loss, refit_actual_float32_MSE=after_loss,
        fit_loss_reduction_percent=100*(1-after_loss/before_loss) if before_loss > 0 else None,
        unknown_proposed_rows=int(((choice >= 0) & ~data['observed_future']).sum()),
        known_proposed_rows=int(((choice >= 0) & data['observed_future']).sum()),
        zero_proposal_queries=sum(not c['allowed'][g].any() for g in c['groups']),
        frozen_representation_sha256=digest(features['design']),
        target_solver_sha256=digest(y), empirical_weights_sha256=digest(w))
    state.update(diagnostics=diagnostics, seconds=time.monotonic()-started,
        optimizer_state=None, independent_confirmation=False, calibrated_safety_bound=False)
    guard(); path.parent.mkdir(parents=True, exist_ok=True); base.save_checkpoint(path, state)
    heartbeat(state='readout_fit_saved', **diagnostics)
    return base.read_checkpoint(path)


def exact_refit(a, b):
    for key in set(a)-{'seconds'}:
        base.exact(a[key], b[key])


predict = parent.predict
