"""Positive mean-preserving harm slopes fitted to a squared score-cost surrogate."""
import numpy as np
from src.world_model import m3w_positive_harm as positive

predict = positive.predict
HARM = positive.HARM


def harm_scales(scale, score_rms):
    sigma = float(scale)*np.asarray(score_rms, float)
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Three positive frozen signed-score scales required')
    return np.array([np.sqrt(1.5/(1/sigma[0]**2+1/sigma[1]**2)), sigma[2]*np.sqrt(1.5)])


def score_equivalent_targets(target, fixed_prediction, score_rms):
    y, p, sigma = np.asarray(target, float), np.asarray(fixed_prediction, float), np.asarray(score_rms, float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 5 or sigma.shape != (3,):
        raise ValueError('Five aligned moment labels and frozen leaf predictions required')
    a, b = 1/sigma[0]**2, 1/sigma[1]**2
    return np.column_stack((y[:, 1]+(a*(p[:, 0]-y[:, 0])+.02*b*(p[:, 2]-y[:, 2]))/(a+b),
                            y[:, 4]+.02*(p[:, 3]-y[:, 3])))


def cost_leaf(leaves, q, target, weights, cost_scales, *, squared_target=None, penalty=1., max_iter=128, tolerance=1e-7):
    q, y, w, scales = [np.asarray(v, float) for v in (q, target, weights, cost_scales)]
    if (q.shape != (len(w), 7) or y.shape != (len(w), 2) or not len(w)
            or not all(np.isfinite(v).all() for v in (q, y, w, scales)) or (y < 0).any()
            or (w <= 0).any() or len(leaves) != len(w) or scales.shape != (2,) or (scales <= 0).any()
            or penalty <= 0 or max_iter < 1 or tolerance <= 0):
        raise ValueError('Known finite nonnegative harm and positive training weights/scales required')
    t = y if squared_target is None else np.asarray(squared_target, float)
    if t.shape != y.shape or not np.isfinite(t).all(): raise ValueError('Finite equivalent score targets required')
    nodes, inv = np.unique(leaves, return_inverse=True); mass = np.bincount(inv, weights=w)
    wn = w/mass[inv]; wm = np.bincount(inv, weights=wn)
    def sums(v): return np.bincount(inv, weights=v, minlength=len(nodes))
    means = np.column_stack([sums(wn*q[:, j])/wm for j in range(7)])
    z = q-means[inv]; mu = np.column_stack([sums(wn*y[:, j])/wm for j in range(2)])
    coefficients, norms, counts, losses, gradients, preservation = [], [], [], [], [], []
    for channel in range(2):
        beta = np.zeros((len(nodes), 7)); iterations = np.zeros(len(nodes), np.int64)
        active_target = mu[:, channel] > 0; sigma = scales[channel]
        def quantities(value, derivative=False):
            tilted, normalizer = positive._distribution(value, z, inv, wn, wm)
            pred = mu[inv, channel]*tilted*wm[inv]/wn
            residual = (pred-t[:, channel])/sigma
            loss = .5*sums(wn*residual**2)/wm+.5*penalty*np.sum(value*value, 1)
            if not derivative: return loss
            first = np.column_stack([sums(tilted*z[:, j]) for j in range(7)])
            jac = pred[:, None]/sigma*(z-first[inv])
            grad = np.column_stack([sums(wn*residual*jac[:, j])/wm for j in range(7)])+penalty*value
            return loss, grad, jac, normalizer
        before = quantities(beta)
        for iteration in range(max_iter):
            value, gradient, jac, _ = quantities(beta, True)
            active = active_target & (np.max(np.abs(gradient), 1) > tolerance)
            if not active.any(): break
            hessian = np.empty((len(nodes), 7, 7))
            for j in range(7):
                for k in range(j, 7):
                    hessian[:, j, k] = hessian[:, k, j] = sums(wn*jac[:, j]*jac[:, k])/wm
            hessian += penalty*np.eye(7)
            step = np.zeros_like(beta)
            step[active] = np.linalg.solve(hessian[active], gradient[active, :, None])[..., 0]
            dot = np.sum(gradient*step, 1); alpha = active.astype(float); pending = active.copy(); candidate = beta.copy()
            for _ in range(24):
                proposal = beta-alpha[:, None]*step; loss = quantities(proposal)
                accept = pending & (loss <= value-1e-4*alpha*dot+1e-12)
                candidate[accept] = proposal[accept]; pending[accept] = False
                if not pending.any(): break
                alpha[pending] *= .5
            if pending.any(): raise RuntimeError('Squared cost line search did not converge')
            beta = candidate; iterations[active] += 1
        after, gradient, _, normalizer = quantities(beta, True)
        maximum = float(np.max(np.abs(gradient[active_target]))) if active_target.any() else 0.
        if maximum > tolerance: raise RuntimeError(f'Squared cost fit not converged: gradient={maximum}')
        beta[~active_target] = 0.; normalizer[~active_target] = 0.
        log_rate = np.einsum('ni,ni->n', z, beta[inv])-normalizer[inv]
        if np.abs(log_rate).max() > 60: raise RuntimeError('Training exceeded fixed numerical range')
        error = float(np.max(np.abs(sums(wn*np.exp(log_rate))/wm-1)))
        assert error < 1e-9 and np.all(after <= before+1e-9)
        coefficients.append(beta); norms.append(normalizer); counts.append(iterations)
        losses.append(dict(before=float(np.average(before, weights=mass)), after=float(np.average(after, weights=mass))))
        gradients.append(maximum); preservation.append(error)
    return dict(nodes=nodes, means=means, coef=np.stack(coefficients, 1), log_normalizer=np.column_stack(norms),
        iterations=np.column_stack(counts), zero_target=mu == 0, target_means=mu, mass=mass,
        loss=losses, maximum_gradient=max(gradients), train_relative_mean_error=max(preservation))


def fit(state, x, envelope, targets, past_quality, sites, recordings, frames, *, settings):
    pr = state['preprocess']; f = positive.base.forest
    f.core.exact(pr, f.core.preprocess(x, envelope, targets, sites, recordings, frames, training_site=pr['training_site']))
    z, _ = f.causal_inputs(x, envelope, pr); known = np.isfinite(targets).all(1)
    w, _ = f.core.weights(sites, recordings, frames, known)
    hashes = dict(features=f.fingerprint(z[known]), targets=f.fingerprint(f.transformed_targets(targets[known], pr)),
        weights=f.fingerprint(w[known]), known_mask=f.fingerprint(known))
    assert hashes == state['input_hashes']
    q = np.asarray(past_quality, float)
    if q.shape != (len(targets), 7) or not np.isfinite(q).all(): raise ValueError('Seven aligned past-quality features required')
    mean = w@q; std = np.sqrt(w@((q-mean)**2)).clip(1e-6)
    qq = positive.base.standardize(q, mean, std)[known]; target = targets[known][:, HARM]
    scales = harm_scales(pr['scale'], pr['rms'][5:])
    chunks = {k: [] for k in ('nodes', 'means', 'coef', 'log_normalizer', 'iterations', 'zero_target')}
    offsets = [0]; losses = []; gradients = []; mean_errors = []
    for tree in state['model'].estimators_:
        leaves = tree.apply(z[known])
        fixed = tree.tree_.value[leaves, :, 0]/f.FACTORS*pr['rms']*pr['scale']
        effective = score_equivalent_targets(targets[known], fixed[:, :5], pr['rms'][5:])
        a = cost_leaf(leaves, qq, target, w[known], scales, squared_target=effective, **settings)
        raw = tree.tree_.value[a['nodes'], :, 0]/f.FACTORS*pr['rms']*pr['scale']
        np.testing.assert_allclose(a['target_means'], raw[:, HARM], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(a['mass'], tree.tree_.weighted_n_node_samples[a['nodes']], rtol=2e-6, atol=1e-10)
        for key in chunks: chunks[key].append(a[key])
        offsets.append(offsets[-1]+len(a['nodes']))
        losses.append(a['loss']); gradients.append(a['maximum_gradient']); mean_errors.append(a['train_relative_mean_error'])
    return dict(**{k: np.concatenate(v) for k, v in chunks.items()}, offsets=np.array(offsets, np.int64),
        quality_mean=mean, quality_std=std, settings=settings, input_hashes=hashes, cost_scales=scales,
        past_quality_hash=f.fingerprint(q), known_train_rows=int(known.sum()), unknown_train_rows=int((~known).sum()),
        original_leaves_reconstructed=offsets[-1], maximum_gradient=max(gradients), train_relative_mean_error=max(mean_errors),
        zero_target_leaves_by_channel=np.concatenate(chunks['zero_target']).sum(0).tolist(),
        max_iterations=int(np.concatenate(chunks['iterations']).max()), loss_name='squared_raw_score_surrogate_plus_fixed_ridge',
        training_loss={k: np.mean([[v[j][k] for j in range(2)] for v in losses], 0).tolist() for k in ('before', 'after')})


def evaluate(state, predictions, targets, envelope, moving, support, sites, recordings, frames, ids):
    base = positive.base
    if set(predictions) != {'original', 'additive', 'poisson', 'cost'}: raise ValueError('All four registered arms required')
    actions = {k: base.eligible(p, moving, support) for k, p in predictions.items()}
    for other in ('original', 'additive', 'poisson'):
        a, b = base.matched(actions[other], actions['cost'], base.forest.core.signed(predictions[other])[:, 0],
            base.forest.core.signed(predictions['cost'])[:, 0], recordings, frames, ids)
        actions[other+'_matched_cost'], actions['cost_matched_'+other] = a, b
        _, inv = np.unique(np.rec.fromarrays([recordings, frames]), return_inverse=True)
        np.testing.assert_array_equal(np.bincount(inv, weights=a), np.bincount(inv, weights=b))
    policies = {k: base.completion_bounds(targets, a, envelope) for k, a in actions.items()}
    scores = {k: base.forest.signed_score(p, targets, state['preprocess'], sites, recordings, frames) for k, p in predictions.items()}
    den = float(targets[np.isfinite(targets).all(1), 2].sum()); contrasts = {}
    for other in ('original', 'additive', 'poisson'):
        contrasts['cost_minus_'+other+'_signed_MSE'] = scores['cost']-scores[other]
        for mode in ('full', 'matched'):
            a = policies[other if mode == 'full' else other+'_matched_cost']
            b = policies['cost' if mode == 'full' else 'cost_matched_'+other]
            contrasts['cost_minus_'+other+'_'+mode+'_utility_percent'] = 100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/den if den > 0 else None
    return dict(policies=policies, scores=scores, contrasts=contrasts, full_known_reference_mass=den), actions


def summarize(rows, cfg):
    out = dict(groups=len(rows), new_cost_fits=len(rows), cached_control_arms=3, new_tree_splits=0,
        new_neural_updates=0, independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for arm in rows[0]['result']['policies']:
        r = [row['result']['policies'][arm] for row in rows]
        risk = [v['easy_selected_risk_upper'] for v in r if v['easy_selected_risk_upper'] is not None]
        out[arm] = dict(selected=sum(v['selected_count'] for v in r), unknown_selected=sum(v['selected_unknown'] for v in r),
            complete_support=sum(v['finite_completion_supported'] for v in r), defined_easy_risk=len(risk),
            violations=sum(v > .02+1e-12 for v in risk), worst_easy_upper=max(risk) if risk else None,
            known_label_violations=sum(v['selected_known_easy_harm_mass']/v['selected_known_easy_reference_mass'] > .02+1e-12
                for v in r if v['selected_known_easy_reference_mass'] > 0))
    for name in rows[0]['result']['contrasts']:
        out[name] = positive.base.interval([(r['source'], r['result']['contrasts'][name]) for r in rows],
            cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    reasons = []
    for other in ('original', 'additive', 'poisson'):
        if out['cost_minus_'+other+'_signed_MSE']['CI95'][1] >= 0: reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full', 'matched'):
            if out['cost_minus_'+other+'_'+mode+'_utility_percent']['CI95'][0] <= 0:
                reasons.append(mode+'_utility_not_supported_vs_'+other)
    c, o = out['cost'], out['original']
    if c['complete_support'] < o['complete_support']: reasons.append('complete_support_reduced')
    if c['violations'] > o['violations']: reasons.append('more_upper_risk_violations')
    if c['worst_easy_upper'] is None or o['worst_easy_upper'] is None or c['worst_easy_upper'] > o['worst_easy_upper']:
        reasons.append('worst_upper_risk_not_preserved')
    out['advance_to_transfer'] = not reasons; out['failure_reasons'] = reasons
    return out
