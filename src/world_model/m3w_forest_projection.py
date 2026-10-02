"""Single-factor forest decoder falsification; no fitted policy or safety claim."""
import numpy as np

from src.world_model.m3w_component_calibration import eligible
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds as bounds


def harm_first(raw, envelope):
    p, env = np.asarray(raw, float).copy(), np.asarray(envelope, float)
    if (env.ndim != 1 or p.shape != (len(env), 5) or not np.isfinite(p).all()
            or (p < -1e-12).any() or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Nonnegative finite moments and causal envelope required')
    p = np.maximum(p, 0.)
    p[:, 1] = np.minimum(p[:, 1], env)
    p[:, 0] = np.minimum(np.minimum(p[:, 0], env-p[:, 1]), p[:, 2])
    p[:, 3] = np.minimum(p[:, 3], p[:, 2])
    p[:, 4] = np.minimum(p[:, 4], p[:, 1])
    return p


def matched_utility(y, original, retained, env, recordings, frames):
    """Conservative gain vs expected uniform thinning within each exact query.

    This is a linear utility expectation, not expected risk of a random policy.
    Entirely unknown targets receive the same negative-envelope completion.
    """
    assert not (retained & ~original).any()
    known = np.isfinite(y).all(1)
    utility = -np.asarray(env).copy()
    utility[known] = y[known, 0]-y[known, 1]
    _, inv = np.unique(np.rec.fromarrays([recordings, frames]), return_inverse=True)
    n = np.bincount(inv, weights=original.astype(int))
    k = np.bincount(inv, weights=retained.astype(int), minlength=len(n))
    u = np.bincount(inv, weights=utility*original, minlength=len(n))
    fraction = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    return float(utility[retained].sum()-(fraction*u).sum())


def interval(pairs, draws, seed):
    sites = sorted({k for k, v in pairs if v is not None})
    if not sites:
        return dict(mean=None, CI95=None, localities=0)
    values = np.array([np.mean([v for k, v in pairs if k == s and v is not None]) for s in sites])
    boot = values[np.random.default_rng(seed).integers(0, len(sites), (draws, len(sites)))].mean(1)
    return dict(mean=float(values.mean()), CI95=np.quantile(boot, [.025, .975]).tolist(),
                localities=len(sites), nominal_exposed_development_only=True)


def audit(raw, projected, new, y, env, moving, support, rec, frames):
    old_action, new_action = [eligible(p, moving, support) for p in (projected, new)]
    np.testing.assert_array_less(new[:, 0], projected[:, 0]+1e-8)
    assert (new[:, 1] >= projected[:, 1]-1e-8).all()
    assert (new[:, 4] >= projected[:, 4]-1e-8).all()
    np.testing.assert_array_equal(new[:, 2:4], projected[:, 2:4])
    assert not (new_action & ~old_action).any()
    removed = old_action & ~new_action
    known = np.isfinite(y).all(1)
    masks = dict(all=np.ones(len(y), bool), parent_selected=old_action,
                 retained=new_action, removed=removed)
    slices = {}
    for name, mask in masks.items():
        at = mask & known
        slices[name] = dict(rows=int(mask.sum()), known=int(at.sum()),
            harm_reduced_by_original_projection=int((mask & (raw[:, 1] > projected[:, 1]+1e-10)).sum()),
            easy_harm_reduced_by_original_projection=int((mask & (raw[:, 4] > projected[:, 4]+1e-10)).sum()),
            binding_envelope=int((mask & (raw[:, :2].sum(1) > env+1e-10)).sum()),
            raw_predicted_mass=raw[mask].sum(0).tolist(),
            original_predicted_mass=projected[mask].sum(0).tolist(),
            harm_first_predicted_mass=new[mask].sum(0).tolist(),
            known_target_mass=y[at].sum(0).tolist(),
            known_harmful_rows=int((at & (y[:, 1] > 0)).sum()))
    before, after, removed_bounds = [bounds(y, a, env) for a in (old_action, new_action, removed)]
    denom = float(y[known, 2].sum())
    return dict(rows=len(y), slices=slices, original=before, harm_first=after, removed=removed_bounds,
        full_known_reference_mass=denom,
        utility_contrast_percent_full_known_reference=(100*(after['selected_net_gain_lower_mass']-
            before['selected_net_gain_lower_mass'])/denom if denom > 0 else None),
        query_count_matched_utility_contrast_percent=(100*matched_utility(y, old_action, new_action,
            env, rec, frames)/denom if denom > 0 else None)), old_action, new_action


def summarize(groups, cfg):
    result = dict(groups=len(groups), source_localities=len({g['source'] for g in groups}),
        result_source='fresh_run_frozen_forest_decoder_control', models='cached_verified',
        new_fits=0, threshold_search=False, independent_roles_read=False, transfer_evaluated=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    for arm in ('original', 'harm_first'):
        values = [g['result'][arm] for g in groups]
        easy = [v['easy_selected_risk_upper'] for v in values if v['easy_selected_risk_upper'] is not None]
        result[arm] = dict(selected=sum(v['selected_count'] for v in values),
            selected_unknown=sum(v['selected_unknown'] for v in values),
            complete_support=sum(v['finite_completion_supported'] for v in values),
            defined_easy_risk=len(easy), easy_upper_violations=sum(v > .02+1e-12 for v in easy),
            worst_easy_upper=max(easy) if easy else None)
    for key in ('utility_contrast_percent_full_known_reference', 'query_count_matched_utility_contrast_percent'):
        result[key] = interval([(g['source'], g['result'][key]) for g in groups],
                               cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    result['projection_counts'] = {name: {k:sum(g['result']['slices'][name][k] for g in groups)
        for k in ('rows','known','harm_reduced_by_original_projection','easy_harm_reduced_by_original_projection',
                  'binding_envelope','known_harmful_rows')} for name in ('all','parent_selected','retained','removed')}
    return result
