"""Non-intervening measurements along exactly reconstructed cost-head training."""
import os
from pathlib import Path
import numpy as np
import torch
from src.world_model import m3w_aux_gradient as gradient
from src.world_model.m3w_native_gain_harm import standardized


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: exact_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): exact_tree(x, y)
    else:
        assert a == b


def match_final(state, original):
    for key in ('model', 'optimizer', 'initial_model', 'draws', 'sampler_rng', 'torch_rng',
                'loss_scales', 'fixed_ids', 'preprocess', 'auxiliary_target', 'settings', 'seed', 'arm', 'step'):
        exact_tree(state[key], original[key])
    # The original first pilot logged step100 in addition to its fixed heartbeat.
    a = {r['step']: r for r in state['trace']}; b = {r['step']: r for r in original['trace']}
    expected = {0, 1, state['step'], *range(state['settings']['heartbeat_every'], state['step']+1,
                                         state['settings']['heartbeat_every'])}
    assert set(a) == expected and len(a) == len(state['trace']) and len(b) == len(original['trace'])
    assert set(a) <= set(b) and set(b)-set(a) <= ({100} if state['step'] > 100 else set())
    for step in a: exact_tree(a[step], b[step])
    return dict(exact_common_trace_steps=len(a), original_extra_pilot_steps=sorted(set(b)-set(a)))


def snapshot(state, path):
    fields = ('model', 'optimizer', 'step', 'settings', 'seed', 'arm', 'identity')
    small = {k: state[k] for k in fields}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        exact_tree(torch.load(path, map_location='cpu', weights_only=False), small)
    else:
        tmp = path.with_suffix('.tmp'); torch.save(small, tmp); os.replace(tmp, path)


def severity_spec(y, envelope, weights, quantiles):
    known = np.isfinite(y).all(1)
    if y.ndim != 2 or y.shape[1] != 4 or np.any(weights[~known]) or np.any(weights < 0):
        raise ValueError('Known fitting-only cost support required')
    positive = known & (envelope > 0) & (y[:, 3] > 0)
    if not positive.any(): raise ValueError('Positive easy-harm fitting support required')
    values, w = y[positive, 3], weights[positive]
    order = np.argsort(values, kind='stable'); values, w = values[order], w[order]
    cumulative = np.cumsum(w); cumulative /= cumulative[-1]
    indices = np.searchsorted(cumulative, quantiles, side='left').clip(max=len(values)-1)
    return dict(quantiles=list(quantiles), edges=values[indices].tolist(),
        method='positive_easy_harm_fitting_equal_locality_weighted_left_cdf',
        positive_rows=int(positive.sum()), held_statistics_used=False)


def strata(y, envelope, spec):
    known = np.isfinite(y).all(1); pos = known & (envelope > 0); h = y[:, 3]
    q50, q90, q99 = spec['edges']
    return dict(all=known, envelope_positive=pos, zero_easy_harm=pos & (h == 0),
        positive_easy_harm=pos & (h > 0),
        severity_0_50=pos & (h > 0) & (h <= q50),
        severity_50_90=pos & (h > q50) & (h <= q90),
        severity_90_99=pos & (h > q90) & (h <= q99),
        severity_99_100=pos & (h > q99))


def measure(model, state, z, target, y, env, sites, spec, event):
    d = torch.tensor(env/state['preprocess']['cost_scale'], dtype=torch.float32)
    scales = torch.tensor(state['loss_scales'], dtype=torch.float32)
    pieces, logits = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), 4096):
            pred, logit = model(z[start:start+4096], d[start:start+4096])
            pieces.append((((pred.double()-target[start:start+4096].double())/scales.double())**2).numpy())
            logits.append(logit.double().numpy())
    errors, l = np.concatenate(pieces), np.concatenate(logits)
    valid = np.isfinite(y).all(1)
    assert np.isfinite(errors[valid]).all()
    masks = strata(y, env, spec)
    local = {}
    for site in sorted(set(sites)):
        group = sites == site; row = {}
        for name, mask in masks.items():
            use = mask & group; n = int(use.sum())
            row[name] = dict(rows=n, cost4=float(errors[use].mean()) if n else None,
                easy_harm=float(errors[use, 3].mean()) if n else None,
                easy_harm_SSE=float(errors[use, 3].sum()) if n else 0.)
        use = group & np.isfinite(event)
        bce = np.logaddexp(0, l[use])-event[use]*l[use]
        row['true_event'] = dict(rows=int(use.sum()), BCE=float(bce.mean()) if use.any() else None)
        # Exact partition accounting prevents missing zero rows or tail mass.
        assert sum(row[k]['rows'] for k in ('severity_0_50', 'severity_50_90', 'severity_90_99', 'severity_99_100')) == row['positive_easy_harm']['rows']
        np.testing.assert_allclose(sum(row[k]['easy_harm_SSE'] for k in
            ('severity_0_50', 'severity_50_90', 'severity_90_99', 'severity_99_100')), row['positive_easy_harm']['easy_harm_SSE'], rtol=1e-12, atol=1e-10)
        local[str(site)] = row
    return local


def gradients(model, state, z, target, env, true_event, samples):
    envelope = torch.tensor(env/state['preprocess']['cost_scale'], dtype=torch.float32)
    scales = torch.tensor(state['loss_scales'], dtype=torch.float32)
    labels = {'actual_training': torch.tensor(state['auxiliary_target'], dtype=torch.float32),
              'true_event': torch.tensor(true_event, dtype=torch.float32)}
    indices = [i for i, (name, _) in enumerate(model.named_parameters()) if name.startswith('network.0.')]
    rows = []
    for ids in samples:
        row = {}
        for name, event in labels.items():
            g, loss = gradient.gradients(model, z[ids], envelope[ids], target[ids], event[ids], scales)
            row[name] = dict(losses=loss,
                cost4_shared=gradient.geometry(gradient.flat(g[0], indices), gradient.flat(g[2], indices)),
                easy_harm_shared=gradient.geometry(gradient.flat(g[1], indices), gradient.flat(g[2], indices)))
        rows.append(row)
    return rows


def tensors(x, y, pr):
    z = torch.from_numpy(standardized(x, pr))
    target = torch.from_numpy(np.where(pr['known'][:, None], y/pr['cost_scale'], 0).astype(np.float32))
    return z, target
