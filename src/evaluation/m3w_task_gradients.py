"""Frozen fitting-batch task gradients and disposable AdamW counterfactuals."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from src.evaluation.m3w_source_gradient_diagnostic import flat_gradient, gradient_relation
from src.world_model.m3w_membership_auxiliary import objective, standardized

COMPONENTS = ('D', 'H', 'D_E', 'H_E')


def fitting_batch(x, env, target, easy, sites, held, state):
    pr = state['preprocess']; ids = np.asarray(state['fixed_ids'])
    n = len(x); known = np.isfinite(target).all(1)
    if (x.ndim != 2 or target.shape != (n, 4) or env.shape != (n,)
            or easy.shape != (n,) or sites.shape != (n,) or held in sites
            or sorted(set(sites)) != pr['training_sites']
            or not np.isfinite(x).all() or not np.isfinite(env).all() or (env < 0).any()
            or not np.array_equal(known, pr['known'])
            or not np.array_equal(np.isnan(target).all(1), ~known)
            or not np.array_equal(np.isfinite(easy), known)
            or not np.isin(easy[known], [0, 1]).all()
            or ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
            or (ids < 0).any() or (ids >= n).any() or not known[ids].all()):
        raise ValueError('Exact supported fitting batch with held locality excluded required')
    np.testing.assert_allclose(target[known, 3], target[known, 1]*easy[known], atol=1e-8)
    scale = pr['cost_scale']
    return (torch.from_numpy(standardized(x[ids], pr)),
            torch.tensor(env[ids]/scale, dtype=torch.float32),
            torch.tensor(target[ids]/scale, dtype=torch.float32),
            torch.tensor(easy[ids], dtype=torch.float32),
            torch.tensor(state['loss_scales'], dtype=torch.float32))


def terms(model, batch):
    z, d, y, e, scales = batch
    pred, logit = model(z, d)
    parts = (((pred-y.detach())/scales)**2).mean(0)/4
    return parts.sum(), F.binary_cross_entropy_with_logits(logit, e.detach()), parts


def values(model, batch):
    with torch.no_grad():
        cost, bce, parts = terms(model, batch)
    return dict(cost=float(cost), BCE=float(bce), **{k: float(v) for k, v in zip(COMPONENTS, parts)})


def virtual_step(model, state, batch, arm):
    clone = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(clone.parameters(), lr=state['settings']['learning_rate'], weight_decay=.0001)
    # load_state_dict may share CPU tensors; isolate momentum and step counters.
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    optimizer.zero_grad(set_to_none=True)
    z, d, y, e, scales = batch
    pred, logit = clone(z, d)
    loss, _ = objective(pred, logit, y, e, scales, arm)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(clone.parameters(), state['settings']['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    delta = torch.cat([(q-p).detach().reshape(-1).double()
                       for p, q in zip(model.parameters(), clone.parameters())]).numpy()
    steps = [int(v['step']) for v in optimizer.state.values()]
    return dict(after=values(clone, batch), preclip_norm=float(norm),
                clip_factor=min(1., state['settings']['gradient_clip']/(float(norm)+1e-6)),
                min_optimizer_step=min(steps), max_optimizer_step=max(steps)), delta


def diagnose(model, state, batch):
    before = values(model, batch)
    last = state['trace'][-1]
    np.testing.assert_allclose(before['cost'], last['cost_loss'], rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(before['BCE'], last['membership_BCE'], rtol=2e-6, atol=1e-8)
    params = list(model.parameters()); shared_size = sum(p.numel() for p in model.network[0].parameters())
    cost, bce, parts = terms(model, batch)
    gradients = {k: flat_gradient(v, params, retain_graph=True)
                 for k, v in zip(COMPONENTS, parts)}
    gradients['cost'] = flat_gradient(cost, params, retain_graph=True)
    gradients['BCE'] = flat_gradient(bce, params)
    np.testing.assert_allclose(sum(gradients[k] for k in COMPONENTS), gradients['cost'], rtol=1e-4, atol=2e-6)
    rel = {k: gradient_relation(gradients['BCE'][:shared_size], gradients[k][:shared_size])
           for k in ('cost', 'H', 'H_E')}
    steps = {}
    for arm in ('cost_only', 'membership_aux'):
        out, delta = virtual_step(model, state, batch, arm)
        out['change'] = {k: out['after'][k]-before[k] for k in before}
        out['first_order_full'] = {k: float(gradients[k]@delta) for k in ('cost', 'H', 'H_E')}
        out['first_order_shared'] = {k: float(gradients[k][:shared_size]@delta[:shared_size]) for k in ('cost', 'H', 'H_E')}
        assert out['min_optimizer_step'] == out['max_optimizer_step'] == state['step']+1
        steps[arm] = out
    return dict(before=before, shared_BCE_relation=rel, virtual_steps=steps,
                auxiliary_minus_cost_step={k: steps['membership_aux']['after'][k]-steps['cost_only']['after'][k] for k in before},
                gradient_additivity_checked=True, fixed_batch_rows=len(batch[0]), checkpoint_step=state['step'])


def population(rows, tolerance=1e-6):
    def dist(a):
        a = [v for v in a if v is not None]
        if not a: return dict(defined=0, median=None, p10=None, p90=None)
        return dict(defined=len(a), median=float(np.median(a)), p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)))
    out = dict(dependent_fitting_views=len(rows), gradients={}, optimizer={})
    for k in ('cost', 'H', 'H_E'):
        rs = [r['shared_BCE_relation'][k] for r in rows]
        out['gradients'][k] = dict(cosine=dist([v['cosine'] for v in rs]),
            negative_cosines=sum(v['cosine'] is not None and v['cosine'] < 0 for v in rs),
            norm_ratio=dist([v['norm_ratio'] for v in rs]))
        deltas = [r['auxiliary_minus_cost_step'][k] for r in rows]
        eps = [max(1e-10, tolerance*abs(r['before'][k])) for r in rows]
        out['optimizer'][k] = dict(auxiliary_minus_cost_step=dist(deltas),
            auxiliary_worse=sum(d > e for d, e in zip(deltas, eps)),
            auxiliary_better=sum(d < -e for d, e in zip(deltas, eps)),
            tied=sum(abs(d) <= e for d, e in zip(deltas, eps)),
            cost_step_worsens=sum(r['virtual_steps']['cost_only']['change'][k] > e for r, e in zip(rows, eps)),
            auxiliary_step_worsens=sum(r['virtual_steps']['membership_aux']['change'][k] > e for r, e in zip(rows, eps)))
    return out
