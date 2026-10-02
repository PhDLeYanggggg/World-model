"""Exact-curvature solver amendment for the same positive squared-cost model."""
import numpy as np
from src.world_model import m3w_cost_aligned_positive_harm as v1

positive, HARM = v1.positive, v1.HARM
predict, evaluate, summarize = v1.predict, v1.evaluate, v1.summarize
harm_scales, score_equivalent_targets = v1.harm_scales, v1.score_equivalent_targets


def quantities(beta, z, inv, wn, wm, mu, target, sigma, penalty, curvature=True):
    count = len(wm)
    def sums(v): return np.bincount(inv, weights=v, minlength=count)
    tilted, normalizer = positive._distribution(beta, z, inv, wn, wm)
    pred = mu[inv]*tilted*wm[inv]/wn; residual = (pred-target)/sigma
    loss = .5*sums(wn*residual**2)/wm+.5*penalty*np.sum(beta*beta, 1)
    if not curvature: return loss
    first = np.column_stack([sums(tilted*z[:, j]) for j in range(7)])
    centered = z-first[inv]
    factor = pred*(pred-target)/sigma**2
    gradient = np.column_stack([sums(wn*factor*centered[:, j])/wm for j in range(7)])+penalty*beta
    hessian = np.empty((count, 7, 7)); curvature_mass = sums(wn*factor)/wm
    square = (2*pred*pred-pred*target)/sigma**2
    for j in range(7):
        for k in range(j, 7):
            cov = sums(tilted*z[:, j]*z[:, k])-first[:, j]*first[:, k]
            entry = sums(wn*square*centered[:, j]*centered[:, k])/wm-curvature_mass*cov
            hessian[:, j, k] = hessian[:, k, j] = entry
    hessian += penalty*np.eye(7)
    return loss, gradient, hessian, normalizer


def cost_leaf(leaves, q, target, weights, cost_scales, *, squared_target=None, penalty=1., max_iter=128, tolerance=1e-7):
    q, y, w, scales = [np.asarray(v, float) for v in (q, target, weights, cost_scales)]
    if (q.shape != (len(w), 7) or y.shape != (len(w), 2) or not len(w)
            or not all(np.isfinite(v).all() for v in (q, y, w, scales)) or (y < 0).any()
            or (w <= 0).any() or len(leaves) != len(w) or scales.shape != (2,) or (scales <= 0).any()
            or penalty <= 0 or max_iter < 1 or tolerance <= 0):
        raise ValueError('Finite known nonnegative harm and positive training weights/scales required')
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
        active_target = mu[:, channel] > 0
        def objective(value, curvature=True):
            return quantities(value, z, inv, wn, wm, mu[:, channel], t[:, channel], scales[channel], penalty, curvature)
        before = objective(beta, False)
        for iteration in range(max_iter):
            value, gradient, hessian, _ = objective(beta)
            active = active_target & (np.max(np.abs(gradient), 1) > tolerance)
            if not active.any(): break
            local_hessian = hessian[active]
            floor = penalty*1e-4
            shift = np.maximum(floor-np.linalg.eigvalsh(local_hessian)[:, 0], 0)
            local_hessian += shift[:, None, None]*np.eye(7)
            step = np.zeros_like(beta)
            step[active] = np.linalg.solve(local_hessian, gradient[active, :, None])[..., 0]
            dot = np.sum(gradient*step, 1); alpha = active.astype(float); pending = active.copy(); candidate = beta.copy()
            for _ in range(32):
                proposal = beta-alpha[:, None]*step; loss = objective(proposal, False)
                accept = pending & (loss <= value-1e-4*alpha*dot+1e-12)
                candidate[accept] = proposal[accept]; pending[accept] = False
                if not pending.any(): break
                alpha[pending] *= .5
            if pending.any(): raise RuntimeError('Exact-curvature cost line search did not converge')
            beta = candidate; iterations[active] += 1
        after, gradient, _, normalizer = objective(beta)
        maximum = float(np.max(np.abs(gradient[active_target]))) if active_target.any() else 0.
        if maximum > tolerance: raise RuntimeError(f'Exact-curvature cost fit not converged: gradient={maximum}')
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
        leaves = tree.apply(z[known]); fixed = tree.tree_.value[leaves, :, 0]/f.FACTORS*pr['rms']*pr['scale']
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
        solver='exact_residual_curvature_train_only_damping',
        training_loss={k: np.mean([[v[j][k] for j in range(2)] for v in losses], 0).tolist() for k in ('before', 'after')})
