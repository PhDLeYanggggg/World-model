"""Frozen cost-component interventions; descriptive, not policy selection."""
import numpy as np
from src.world_model import m3w_past_quality_auxiliary as quality

forest = quality.forest
COMPONENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')


def raw_predictions(state, fit, x, envelope, past_quality):
    z, support = forest.causal_inputs(x, envelope, state['preprocess'])
    q = quality.standardize(past_quality, fit['quality_mean'], fit['quality_std'])
    if len(q) != len(z):
        raise ValueError('Aligned past-only inputs required')
    pr = state['preprocess']
    state['model'].set_params(n_jobs=1)
    raw = state['model'].predict(z) / forest.FACTORS * pr['rms'] * pr['scale']
    delta = np.zeros((len(x), 5))
    for j, tree in enumerate(state['model'].estimators_):
        start, end = fit['offsets'][j:j+2]
        nodes = fit['nodes'][start:end]
        leaves = tree.apply(z)
        at = np.searchsorted(nodes, leaves)
        assert (at < len(nodes)).all()
        np.testing.assert_array_equal(nodes[at], leaves)
        at += start
        delta += np.einsum('ni,nij->nj', q-fit['means'][at], fit['coef'][at])
    delta *= fit['output_scale'] / len(state['model'].estimators_)
    return raw[:, :5], delta, support


def variants(raw, delta, envelope):
    raw, delta = np.asarray(raw, float), np.asarray(delta, float)
    if (raw.ndim != 2 or raw.shape[1] != 5 or delta.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(delta).all()):
        raise ValueError('Five finite frozen cost components required')
    def project(d):
        return forest.project_moments(np.maximum(raw+d, 0), envelope)
    out = dict(original=forest.project_moments(raw, envelope), quality=project(delta))
    for j, name in enumerate(COMPONENTS):
        only = np.zeros_like(delta)
        only[:, j] = delta[:, j]
        out['only_'+name] = project(only)
        without = delta.copy()
        without[:, j] = 0
        out['without_'+name] = project(without)
    # Diagnostic before feasibility projection, never a deployable predictor.
    out['quality_preprojection'] = np.maximum(raw+delta, 0)
    return out


def slack_decomposition(p, y, take, envelope):
    known = np.isfinite(y).all(1)
    at = take & known
    truth, pred = y[at].sum(0), p[at].sum(0)
    out = dict(selected=int(take.sum()), known=int(at.sum()),
               unknown=int((take & ~known).sum()),
               unknown_envelope_mass=float(envelope[take & ~known].sum()),
               known_easy_benefit_mass=float(y[at & (y[:, 3] > 0), 0].sum()),
               known_truth_moments=truth.tolist(), known_prediction_moments=pred.tolist())
    for name, h, r in (('all', 1, 2), ('easy', 4, 3)):
        actual = truth[h]-.02*truth[r]
        predicted = pred[h]-.02*pred[r]
        harm_error = truth[h]-pred[h]
        reference_error = .02*(pred[r]-truth[r])
        np.testing.assert_allclose(actual, predicted+harm_error+reference_error, atol=1e-8, rtol=1e-10)
        out[name] = dict(observed_excess_mass=float(actual), predicted_excess_mass=float(predicted),
                         harm_underestimate_mass=float(harm_error),
                         reference_inflation_budget_mass=float(reference_error))
    return out


def evaluate(predictions, y, envelope, moving, support, recordings, frames, ids):
    actions = {k: quality.eligible(p, moving, support) for k, p in predictions.items()}
    old = actions['original']
    old_score = forest.core.signed(predictions['original'])
    known = np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    if den <= 0:
        raise ValueError('Defined full known reference required for this fixed diagnostic')
    out = {}
    for name, p in predictions.items():
        take = actions[name]
        a, b = quality.matched(old, take, old_score[:, 0], forest.core.signed(p)[:, 0],
                               recordings, frames, ids)
        _, inv = np.unique(np.rec.fromarrays([recordings, frames]), return_inverse=True)
        np.testing.assert_array_equal(np.bincount(inv, weights=a), np.bincount(inv, weights=b))
        full = quality.completion_bounds(y, take, envelope)
        am = quality.completion_bounds(y, a, envelope)
        bm = quality.completion_bounds(y, b, envelope)
        cohort = {k: slack_decomposition(p, y, mask, envelope) for k, mask in
                  dict(added=take & ~old, removed=old & ~take, retained=take & old).items()}
        added = take & ~old
        assert sum(cohort[k]['selected'] for k in ('added', 'retained')) == full['selected_count']
        out[name] = dict(full=full, original_matched=am, variant_matched=bm, cohorts=cohort,
                         original_failed_constraints_on_added=dict(
                             nonpositive_utility=int((added & (old_score[:, 0] <= 0)).sum()),
                             all_risk=int((added & (old_score[:, 1] > 0)).sum()),
                             easy_risk=int((added & (old_score[:, 2] > 0)).sum())),
                         matched_utility_difference_percent=100*(bm['selected_net_gain_lower_mass']-
                             am['selected_net_gain_lower_mass'])/den)
    old_mass = out['original']['full']['selected_net_gain_lower_mass']
    for row in out.values():
        row['full_utility_difference_percent'] = 100*(row['full']['selected_net_gain_lower_mass']-old_mass)/den
    return dict(arms=out, full_known_reference_mass=den,
                action_hashes={k: forest.fingerprint(a) for k, a in actions.items()},
                quality_projection_removed_actions=int((actions['quality_preprojection'] & ~actions['quality']).sum()),
                quality_projection_added_actions=int((~actions['quality_preprojection'] & actions['quality']).sum()))
