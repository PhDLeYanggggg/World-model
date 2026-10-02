"""Mean-preserving positive conditional harm, not a calibrated safety bound."""
import numpy as np
from src.world_model import m3w_past_quality_auxiliary as base

HARM = (1, 4)


def _distribution(beta, z, inv, w, mass):
    eta = np.einsum('ni,ni->n', z, beta[inv])
    peak = np.full(len(mass), -np.inf)
    np.maximum.at(peak, inv, eta)
    unnormalized = w*np.exp(eta-peak[inv])
    total = np.bincount(inv, weights=unnormalized, minlength=len(mass))
    log_normalizer = peak+np.log(total)-np.log(mass)
    return unnormalized/total[inv], log_normalizer


def positive_leaf(leaves, q, target, weights, *, penalty=1., max_iter=32, tolerance=1e-7):
    q, y, w = np.asarray(q, float), np.asarray(target, float), np.asarray(weights, float)
    if (q.shape != (len(w), 7) or y.shape != (len(w), 2) or not len(w)
            or not np.isfinite(q).all() or not np.isfinite(y).all() or (y < 0).any()
            or not np.isfinite(w).all() or (w <= 0).any() or len(leaves) != len(w)
            or penalty <= 0 or max_iter < 1 or tolerance <= 0):
        raise ValueError('Finite known rows, nonnegative harm and positive weights required')
    nodes, inv = np.unique(leaves, return_inverse=True)
    mass = np.bincount(inv, weights=w)
    wn = w/mass[inv]
    wm = np.bincount(inv, weights=wn)
    def sums(v): return np.bincount(inv, weights=v, minlength=len(nodes))
    means = np.column_stack([sums(wn*q[:, j])/wm for j in range(7)])
    z = q-means[inv]
    mu = np.column_stack([sums(wn*y[:, j])/wm for j in range(2)])
    coefficients, norms, counts, losses, gradients, preservation = [], [], [], [], [], []
    for channel in range(2):
        active_target = mu[:, channel] > 0
        target_mass = sums(wn*y[:, channel])
        target_weights = np.divide(wn*y[:, channel], target_mass[inv],
                                   out=np.zeros(len(w)), where=target_mass[inv] > 0)
        target_mean = np.column_stack([sums(target_weights*z[:, j]) for j in range(7)])
        ratio = np.divide(y[:, channel], mu[inv, channel], out=np.zeros(len(w)),
                          where=mu[inv, channel] > 0)
        log_ratio = np.zeros(len(w)); positive = ratio > 0
        log_ratio[positive] = np.log(ratio[positive])
        deviance0 = np.maximum(sums(wn*ratio*log_ratio)/wm, 0.)
        beta = np.zeros((len(nodes), 7)); iterations = np.zeros(len(nodes), np.int64)
        def objective(value):
            _, normalizer = _distribution(value, z, inv, wn, wm)
            return normalizer-np.sum(target_mean*value, 1)+.5*penalty*np.sum(value*value, 1)
        for iteration in range(max_iter):
            tilted, normalizer = _distribution(beta, z, inv, wn, wm)
            first = np.column_stack([sums(tilted*z[:, j]) for j in range(7)])
            gradient = first-target_mean+penalty*beta
            active = active_target & (np.max(np.abs(gradient), 1) > tolerance)
            if not active.any(): break
            hessian = np.empty((len(nodes), 7, 7))
            for j in range(7):
                for k in range(j, 7):
                    hessian[:, j, k] = hessian[:, k, j] = sums(tilted*z[:, j]*z[:, k])-first[:, j]*first[:, k]
            hessian += penalty*np.eye(7)
            step = np.zeros_like(beta)
            step[active] = np.linalg.solve(hessian[active], gradient[active, :, None])[..., 0]
            value = objective(beta); dot = np.sum(gradient*step, 1)
            alpha = active.astype(float); pending = active.copy(); candidate = beta.copy()
            for _ in range(24):
                proposal = beta-alpha[:, None]*step
                new_value = objective(proposal)
                accept = pending & (new_value <= value-1e-4*alpha*dot+1e-12)
                candidate[accept] = proposal[accept]; pending[accept] = False
                if not pending.any(): break
                alpha[pending] *= .5
            if pending.any(): raise RuntimeError('Conditional deviance line search did not converge')
            beta = candidate; iterations[active] += 1
        tilted, normalizer = _distribution(beta, z, inv, wn, wm)
        first = np.column_stack([sums(tilted*z[:, j]) for j in range(7)])
        gradient = first-target_mean+penalty*beta
        maximum = float(np.max(np.abs(gradient[active_target]))) if active_target.any() else 0.
        if maximum > tolerance:
            raise RuntimeError(f'Conditional deviance fit not converged: gradient={maximum}')
        beta[~active_target] = 0.; normalizer[~active_target] = 0.
        log_rate = np.einsum('ni,ni->n', z, beta[inv])-normalizer[inv]
        if np.abs(log_rate).max() > 60:
            raise RuntimeError('Training numerical range exceeds the fixed inference guard')
        factor = np.exp(log_rate)
        error = float(np.max(np.abs(sums(wn*factor)/wm-1)))
        assert error < 1e-9
        after = deviance0+objective(beta)
        assert np.all(after <= deviance0+1e-9) and (after >= -1e-9).all()
        coefficients.append(beta); norms.append(normalizer); counts.append(iterations)
        losses.append(dict(before=float(np.average(deviance0, weights=mass)),
                           after=float(np.average(after, weights=mass))))
        gradients.append(maximum); preservation.append(error)
    return dict(nodes=nodes, means=means, coef=np.stack(coefficients, 1),
                log_normalizer=np.column_stack(norms), iterations=np.column_stack(counts),
                zero_target=mu == 0, target_means=mu, mass=mass,
                loss=losses, maximum_gradient=max(gradients),
                train_relative_mean_error=max(preservation))


def fit(state, x, envelope, targets, past_quality, sites, recordings, frames, *, settings):
    pr = state['preprocess']; f = base.forest
    f.core.exact(pr, f.core.preprocess(x, envelope, targets, sites, recordings, frames,
                                      training_site=pr['training_site']))
    z, _ = f.causal_inputs(x, envelope, pr)
    known = np.isfinite(targets).all(1)
    w, _ = f.core.weights(sites, recordings, frames, known)
    input_hashes = dict(features=f.fingerprint(z[known]),
        targets=f.fingerprint(f.transformed_targets(targets[known], pr)),
        weights=f.fingerprint(w[known]), known_mask=f.fingerprint(known))
    assert input_hashes == state['input_hashes']
    q = np.asarray(past_quality, float)
    if q.shape != (len(targets), 7) or not np.isfinite(q).all():
        raise ValueError('Seven aligned past-quality features required')
    mean = w@q; std = np.sqrt(w@((q-mean)**2)).clip(1e-6)
    qq = base.standardize(q, mean, std)[known]
    target = targets[known][:, HARM]
    chunks = {k: [] for k in ('nodes', 'means', 'coef', 'log_normalizer', 'iterations', 'zero_target')}
    offsets = [0]; losses = []; gradients = []; mean_errors = []
    for tree in state['model'].estimators_:
        a = positive_leaf(tree.apply(z[known]), qq, target, w[known], **settings)
        original = tree.tree_.value[a['nodes'], :, 0]/f.FACTORS*pr['rms']*pr['scale']
        np.testing.assert_allclose(a['target_means'], original[:, HARM], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(a['mass'], tree.tree_.weighted_n_node_samples[a['nodes']], rtol=2e-6, atol=1e-10)
        for key in chunks: chunks[key].append(a[key])
        offsets.append(offsets[-1]+len(a['nodes']))
        losses.append(a['loss']); gradients.append(a['maximum_gradient']); mean_errors.append(a['train_relative_mean_error'])
    return dict(**{k: np.concatenate(v) for k, v in chunks.items()}, offsets=np.array(offsets, np.int64),
                quality_mean=mean, quality_std=std, settings=settings, input_hashes=input_hashes,
                past_quality_hash=f.fingerprint(q), known_train_rows=int(known.sum()),
                unknown_train_rows=int((~known).sum()), original_leaves_reconstructed=offsets[-1],
                maximum_gradient=max(gradients), train_relative_mean_error=max(mean_errors),
                zero_target_leaves_by_channel=np.concatenate(chunks['zero_target']).sum(0).tolist(),
                max_iterations=int(np.concatenate(chunks['iterations']).max()),
                training_loss={k: np.mean([[v[j][k] for j in range(2)] for v in losses], 0).tolist()
                               for k in ('before', 'after')})


def predict(state, fitted, x, envelope, past_quality):
    f = base.forest; z, support = f.causal_inputs(x, envelope, state['preprocess'])
    q = base.standardize(past_quality, fitted['quality_mean'], fitted['quality_std'])
    if len(q) != len(z): raise ValueError('Aligned causal inputs required')
    pr = state['preprocess']; state['model'].set_params(n_jobs=1)
    raw = state['model'].predict(z)/f.FACTORS*pr['rms']*pr['scale']
    positive = np.zeros((len(x), 8)); clipped = 0
    for j, tree in enumerate(state['model'].estimators_):
        start, end = fitted['offsets'][j:j+2]
        nodes = fitted['nodes'][start:end]; leaves = tree.apply(z)
        at = np.searchsorted(nodes, leaves); assert (at < len(nodes)).all()
        np.testing.assert_array_equal(nodes[at], leaves); at += start
        score = np.einsum('ni,nji->nj', q-fitted['means'][at], fitted['coef'][at])-fitted['log_normalizer'][at]
        clipped += int((np.abs(score) > 60).sum())
        rate = np.exp(np.clip(score, -60, 60))
        values = tree.tree_.value[leaves, :, 0].copy()
        values[:, HARM] *= rate
        positive += values
    positive /= len(state['model'].estimators_)
    positive = positive/f.FACTORS*pr['rms']*pr['scale']
    np.testing.assert_array_equal(positive[:, (0, 2, 3)], raw[:, (0, 2, 3)])
    original = f.project_moments(raw[:, :5], envelope)
    output = f.project_moments(positive[:, :5], envelope)
    return original, output, support, dict(exponential_guard_coordinates=clipped,
                projected_benefit_changed=int((output[:, 0] != original[:, 0]).sum()),
                zero_predicted_harm=int((output[:, 1] == 0).sum()),
                zero_predicted_easy_harm=int((output[:, 4] == 0).sum()))


def evaluate(state, predictions, targets, envelope, moving, support, sites, recordings, frames, ids):
    # Reuse the tested three-arm reader with explicit role aliases, not placebo semantics.
    r, a = base.evaluate(state, dict(original=predictions['original'], quality=predictions['positive'],
        placebo=predictions['additive']), targets, envelope, moving, support, sites, recordings, frames, ids)
    names = dict(original='original', quality='positive', placebo='additive',
                 original_matched_quality='original_matched_positive', quality_matched_original='positive_matched_original',
                 placebo_matched_quality='additive_matched_positive', quality_matched_placebo='positive_matched_additive')
    contrasts = {f'positive_minus_{other}_{metric}': r['contrasts'][f'quality_minus_{old}_{metric}']
                 for other, old in (('original', 'original'), ('additive', 'placebo'))
                 for metric in ('signed_MSE', 'full_utility_percent', 'matched_utility_percent')}
    return dict(policies={names[k]: v for k, v in r['policies'].items()},
                scores={names[k]: v for k, v in r['scores'].items()}, contrasts=contrasts,
                full_known_reference_mass=r['full_known_reference_mass']), {names[k]: v for k, v in a.items()}


def summarize(groups, cfg):
    out = dict(groups=len(groups), new_conditional_harm_fits=len(groups),
               cached_additive_controls=len(groups), new_tree_splits=0, new_neural_updates=0,
               independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for name in groups[0]['result']['policies']:
        rows = [g['result']['policies'][name] for g in groups]
        risk = [r['easy_selected_risk_upper'] for r in rows if r['easy_selected_risk_upper'] is not None]
        out[name] = dict(selected=sum(r['selected_count'] for r in rows),
            unknown_selected=sum(r['selected_unknown'] for r in rows),
            complete_support=sum(r['finite_completion_supported'] for r in rows),
            defined_easy_risk=len(risk), violations=sum(v > .02+1e-12 for v in risk),
            known_label_violations=sum(r['selected_known_easy_harm_mass']/r['selected_known_easy_reference_mass'] > .02+1e-12
                for r in rows if r['selected_known_easy_reference_mass'] > 0),
            worst_easy_upper=max(risk) if risk else None)
    for key in groups[0]['result']['contrasts']:
        out[key] = base.interval([(g['source'], g['result']['contrasts'][key]) for g in groups],
                                cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    reasons = []
    for other in ('original', 'additive'):
        ci = out['positive_minus_'+other+'_signed_MSE']['CI95']
        if ci is None or ci[1] >= 0: reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full', 'matched'):
            ci = out['positive_minus_'+other+'_'+mode+'_utility_percent']['CI95']
            if ci is None or ci[0] <= 0: reasons.append(mode+'_utility_not_supported_vs_'+other)
    p, o = out['positive'], out['original']
    if p['complete_support'] < o['complete_support']: reasons.append('complete_support_reduced')
    if p['violations'] > o['violations']: reasons.append('more_upper_risk_violations')
    if p['known_label_violations'] > o['known_label_violations']: reasons.append('more_known_risk_violations')
    if p['worst_easy_upper'] is None or o['worst_easy_upper'] is None or p['worst_easy_upper'] > o['worst_easy_upper']:
        reasons.append('worst_risk_not_preserved')
    out['gate_failure_reasons'] = reasons
    out['advance_to_transfer'] = not reasons
    return out
