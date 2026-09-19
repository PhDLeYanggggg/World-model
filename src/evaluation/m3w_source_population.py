"""Source-population diagnostics without changing the registered primary metric."""
from __future__ import annotations

import numpy as np

from src.data_unification.m3w_causal_recordings import BASELINES
from src.world_model.m3w_track_event_sampling import EVENT_NAMES, training_event_labels


def scale_floor_mask(values):
    # The legacy cache stores a float32 causal feature in a float64 scale array.
    return np.isin(values, [.001, float(np.float32(.001))])


def baseline_errors(geometry, target, valid):
    g, y, mask = np.asarray(geometry), np.asarray(target), np.asarray(valid)
    n = len(g)
    if g.shape != (n, 476) or y.shape != (n, 12, 2) or mask.shape != (n, 12):
        raise ValueError('Frozen 476-feature, 8-to-12 schema required')
    if mask.dtype != bool or not np.isfinite(g).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Finite observed features, available labels and boolean masks required')
    if np.any(g[:, 16:24] > 0):
        raise ValueError('History may not extend beyond the query')
    neighbor_mask = g[:, 230:294].reshape(n, 8, 8)
    neighbor_time = g[:, 166:230].reshape(n, 8, 8)
    if not np.isin(neighbor_mask, [0, 1]).all() or np.any(neighbor_time[neighbor_mask.astype(bool)] > 0):
        raise ValueError('Only masked past neighbors may be inputs')
    rollouts = g[:, 308:].reshape(n, len(BASELINES), 12, 2).astype(np.float64)
    safe_y = np.where(mask[..., None], y, 0).astype(np.float64)
    distance = np.linalg.norm(rollouts-safe_y[:, None], axis=-1)
    count = mask.sum(1)
    ade = np.divide(np.where(mask[:, None], distance, 0).sum(2), count[:, None],
                    out=np.full((n, len(BASELINES)), np.nan), where=count[:, None] > 0)
    fde = np.where(mask[:, -1, None], distance[:, :, -1], np.nan)
    events = np.full(n, -1, np.int64)
    complete = mask.all(1)
    if complete.any():
        events[complete] = training_event_labels(g[complete], y[complete])
    return ade, fde, events


def scene_mean(values, sites):
    v, s = np.asarray(values, float), np.asarray(sites)
    if not len(v) or len(v) != len(s) or not np.isfinite(v).all():
        raise ValueError('Nonempty finite supported rows and aligned sites required')
    return np.mean([v[s == key].mean(0) for key in np.unique(s)], axis=0)


def safe_gain(error, reference):
    return None if reference <= 0 else float(100*(1-error/reference))


def baseline_table(ade, fde, sites, scale):
    ade, fde, sites, scale = map(np.asarray, (ade, fde, sites, scale))
    if not len(ade):
        return dict(rows=0)
    mean = scene_mean(ade, sites)
    native = scene_mean(ade*scale[:, None], sites)
    last = np.isfinite(fde).all(1)
    final_mean = scene_mean(fde[last], sites[last]) if last.any() else None
    final_native = scene_mean(fde[last]*scale[last, None], sites[last]) if last.any() else None
    strongest = int(mean.argmin())
    oracle = float(scene_mean(ade.min(1), sites))
    return dict(rows=len(ade), sites=len(np.unique(sites)), final_label_rows=int(last.sum()),
        baseline_metrics={name:dict(normalized_ade=float(mean[i]), native_pixel_ade_diagnostic=float(native[i]),
            normalized_fde=None if final_mean is None else float(final_mean[i]),
            native_pixel_fde_diagnostic=None if final_native is None else float(final_native[i]))
            for i, name in enumerate(BASELINES)},
        same_cohort_best_fixed_baseline_diagnostic=BASELINES[strongest],
        oracle_normalized_ade_diagnostic=oracle,
        oracle_headroom_over_same_cohort_best_percent=safe_gain(oracle, mean[strongest]),
        oracle_headroom_over_cv_percent=safe_gain(oracle, mean[1]),
        oracle_is_future_informed_and_not_a_model=True)


def complement_selected_baseline(ade, sites):
    """Choose one of seven fixed causal baselines using other source sites only."""
    ade, sites = np.asarray(ade), np.asarray(sites)
    if len(np.unique(sites)) < 2:
        raise ValueError('At least two source sites required')
    chosen = np.zeros(len(ade), np.int64)
    folds = {}
    for site in np.unique(sites):
        held = sites == site
        train_mean = scene_mean(ade[~held], sites[~held])
        k = int(train_mean.argmin())
        chosen[held] = k
        folds[str(site)] = dict(selected=BASELINES[k], selection_sites=sorted(set(sites[~held])),
            train_ade={name:float(v) for name, v in zip(BASELINES, train_mean)},
            held_ade=float(ade[held, k].mean()), held_cv_ade=float(ade[held, 1].mean()),
            held_gain_over_cv_percent=safe_gain(ade[held, k].mean(), ade[held, 1].mean()))
    error = ade[np.arange(len(ade)), chosen]
    return dict(folds=folds, normalized_ade=float(scene_mean(error, sites)),
        cv_normalized_ade=float(scene_mean(ade[:, 1], sites)),
        gain_over_cv_percent=safe_gain(scene_mean(error, sites), scene_mean(ade[:, 1], sites)),
        oracle_headroom_percent=safe_gain(scene_mean(ade.min(1), sites), scene_mean(error, sites)))


def contribution(error, subset, sites):
    """Share of the full cohort's equal-site error, not reweighted subset error."""
    error, subset, sites = np.asarray(error), np.asarray(subset, bool), np.asarray(sites)
    denominator = float(scene_mean(error, sites))
    numerator = float(scene_mean(np.where(subset, error, 0), sites))
    return dict(equal_site_error_contribution=numerator,
        percent_of_total=None if denominator == 0 else float(100*numerator/denominator))


def window_support(tracks, frames, events):
    tracks, frames, events = map(np.asarray, (tracks, frames, events))
    order = np.lexsort((frames, tracks))
    t, f, e = tracks[order], frames[order], events[order]
    if not len(t):
        return dict(rows=0, tracks=0, annotation_category_runs=0, disjoint_spans=0)
    new_track = np.r_[True, t[1:] != t[:-1]]
    boundary = new_track | np.r_[False, (np.diff(f) != 12) | (e[1:] != e[:-1])]
    disjoint, end = 0, -np.inf
    for q, reset in zip(f, new_track):
        if reset:
            end = -np.inf
        if q-84 > end:
            disjoint += 1
            end = q+144
    return dict(rows=len(t), tracks=int(new_track.sum()), annotation_category_runs=int(boundary.sum()),
                disjoint_spans=disjoint, independent_events_claimed=False)


def describe(values):
    a = np.asarray(values, float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
        p90=float(np.quantile(a, .9)), p99=float(np.quantile(a, .99)), maximum=float(a.max()))


def summarize(a):
    supported = a['valid_count'] > 0
    complete = a['valid_count'] == 12
    n = len(supported)
    report = dict(rows=n, recordings=len(np.unique(a['recordings'])), sites=len(np.unique(a['sites'])),
        complete=int(complete.sum()), partial=int((supported & ~complete).sum()), absent=int((~supported).sum()),
        static_history=int(a['static_history'].sum()), scale=describe(a['scale']),
        scale_floor_rows=int(a['scale_floor'].sum()), cohorts={}, events={})
    for name, mask in [('supported_masked', supported), ('complete', complete)]:
        report['cohorts'][name] = baseline_table(a['ade'][mask], a['fde'][mask], a['sites'][mask], a['scale'][mask])
        if mask.any():
            report['cohorts'][name]['scale_floor_cv_error_share'] = contribution(a['ade'][mask, 1], a['scale_floor'][mask], a['sites'][mask])
            report['cohorts'][name]['scale_floor_native_cv_error_share'] = contribution(a['ade'][mask, 1]*a['scale'][mask], a['scale_floor'][mask], a['sites'][mask])
    for k, name in enumerate(EVENT_NAMES):
        mask = a['event'] == k
        result = baseline_table(a['ade'][mask], a['fde'][mask], a['sites'][mask], a['scale'][mask])
        result.update(window_support(a['tracks'][mask], a['frames'][mask], a['event'][mask]))
        result['cv_normalized_contribution'] = contribution(a['ade'][complete, 1], mask[complete], a['sites'][complete])
        result['cv_native_contribution_diagnostic'] = contribution(a['ade'][complete, 1]*a['scale'][complete], mask[complete], a['sites'][complete])
        result['cv_ade_distribution'] = describe(a['ade'][mask, 1])
        result['scale'] = describe(a['scale'][mask])
        report['events'][name] = result
    report['complete_window_support'] = window_support(a['tracks'][complete], a['frames'][complete], a['event'][complete])
    return report
