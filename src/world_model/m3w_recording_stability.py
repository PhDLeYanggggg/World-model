"""Nested recording-deletion diagnosis on frozen source forecasts."""
from collections import Counter
import json
import math

import numpy as np

from src.world_model import m3w_selected_set_calibration as selected
from scripts import verify_m3w_selected_set_readout as check

ARMS = ('raw_pool', 'selected_set')


def fit_excluding(z, excluded, arm, cfg):
    if arm not in ARMS:
        raise ValueError('Frozen calibration arm required')
    rec = z['recordings'].astype(str)
    keep = ~np.isin(rec, list(excluded))
    values = {k: z[k][keep] for k in ('p', 'y', 'env', 'moving', 'support')}
    if arm == 'selected_set':
        cal = selected.fit(**values, recordings=rec[keep], mode='joint',
                           rounds=cfg['round_cap'], quantile=cfg['empirical_quantile'])
    else:
        action = selected.parent.eligible(values['p'], values['moving'], values['support'])
        scores = selected.parent.record_scores(values['p'], values['y'], values['env'],
                                                action, rec[keep])
        cal = selected.parent.fit_margin(scores, cfg['empirical_quantile'])
    assert set(cal['recordings']) == set(rec[keep])
    assert not set(cal['recordings']) & set(excluded)
    return cal


def infer_consensus(p, env, moving, support, base, deletions):
    """Inference has no labels. An unestimable deletion disables consensus."""
    original = check.actions(p, env, moving, support, base, 'joint')
    take = original.copy()
    alternatives = [check.actions(p, env, moving, support, c, 'joint') for c in deletions]
    estimable = bool(deletions) and all(c['supported'] for c in deletions)
    if not estimable:
        take[:] = False
    else:
        for action in alternatives:
            take &= action
    assert not (take & ~original).any()
    return original, take, alternatives, estimable


def matched_expectation(y, env, original, retained, recordings):
    """Exact expected lower utility under uniform same-recording count thinning.

    This is linear mass expectation, NOT an expected risk ratio or safety pass.
    """
    if (retained & ~original).any():
        raise ValueError('Retained decisions must be a subset')
    known = np.isfinite(y).all(1)
    reward = np.where(known, y[:, 0]-y[:, 1], -env)
    total, count = [], 0
    for rec in np.unique(recordings):
        ix = (recordings == rec) & original
        n, k = int(ix.sum()), int(((recordings == rec) & retained).sum())
        if k > n:
            raise ValueError('Invalid matched count')
        count += k
        if n:
            total.append(k / n * math.fsum(float(v) for v in reward[ix]))
    return dict(selected_occurrences=count, conservative_utility_expectation=math.fsum(total),
                ratio_or_safety_guarantee=False, matching_unit='whole_recording_not_frame')


def audit_head(z, old, group, cfg):
    rec = z['recordings'].astype(str)
    names = sorted(set(rec))
    known = np.isfinite(z['y']).all(1)
    full_ref = math.fsum(float(v) for v in z['y'][known, 2])
    rows = {}
    for arm in ARMS:
        cache = {}

        def fitted(excluded):
            key = tuple(sorted(excluded))
            if key not in cache:
                cache[key] = fit_excluding(z, key, arm, cfg)
            return cache[key]

        frozen = old['folds'] if arm == 'raw_pool' else next(
            r['folds'] for r in group['result']['rows']
            if r['mode'] == 'joint' and r['role'] == 'source_oof')
        frozen = {f['held_recording']: f['calibration'] for f in frozen}
        assert set(frozen) == set(names)
        original = np.zeros(len(rec), bool)
        retained = np.zeros(len(rec), bool)
        unsupported_removed = np.zeros(len(rec), bool)
        folds = []
        for held in names:
            ix = rec == held
            base = fitted([held])
            assert base == frozen[held], 'Existing OOF calibration must replay exactly'
            nested = [fitted([held, other]) for other in names if other != held]
            b, a, alternatives, estimable = infer_consensus(
                z['p'][ix], z['env'][ix], z['moving'][ix], z['support'][ix], base, nested)
            original[ix], retained[ix] = b, a
            if not estimable:
                unsupported_removed[ix] = b
            switches = [int((x != b).sum()) for x in alternatives]
            folds.append(dict(held_recording=held, calibration_recordings=len(names)-1,
                rows=int(ix.sum()), base_selected=int(b.sum()), retained=int(a.sum()),
                base_component_support=base['component_recording_counts'],
                nested_count=len(nested), nested_supported=sum(c['supported'] for c in nested),
                estimable=estimable, deletion_changed_occurrences=switches,
                deletion_lost_occurrences=[int((b & ~x).sum()) for x in alternatives],
                deletion_added_occurrences=[int((~b & x).sum()) for x in alternatives],
                minimum_nested_component_support=min((min(c['component_recording_counts']) for c in nested), default=0)))
        anchor = old['oof_action_hashes']['joint'] if arm == 'raw_pool' else next(
            r['action_hash'] for r in group['result']['rows']
            if r['mode'] == 'joint' and r['role'] == 'source_oof')
        assert check.array_hash(original) == anchor
        removed = original & ~retained
        bb, aa, rr = (check.scalar_bounds(z['y'], x, z['env']) for x in (original, retained, removed))
        for key in ('selected_count', 'selected_unknown', 'selected_known_benefit_mass',
                    'selected_known_harm_mass', 'selected_unknown_envelope_mass',
                    'selected_known_reference_mass', 'selected_known_easy_reference_mass',
                    'selected_known_easy_harm_mass', 'selected_net_gain_lower_mass'):
            assert math.isclose(bb[key], aa[key]+rr[key], rel_tol=1e-10, abs_tol=1e-10), key
        expected = matched_expectation(z['y'], z['env'], original, retained, rec)
        rows[arm] = dict(parent=bb, consensus=aa, removed=rr, matched=expected, folds=folds,
            parent_hash=anchor, consensus_hash=check.array_hash(retained),
            removed_for_unestimable_deletion=int(unsupported_removed.sum()),
            removed_despite_estimable_deletions=int((removed & ~unsupported_removed).sum()),
            unique_calibrators=len(cache),
            all_nested_statuses=dict(Counter(c.get('status', 'raw_pool') for k, c in cache.items() if len(k) == 2)),
            nested_calibration_digest=check.digest(json.dumps(
                [dict(excluded=list(k), calibration=v) for k, v in sorted(cache.items())], sort_keys=True).encode()))
    return dict(source=group['source'], group=group['identity'], source_recordings=len(names),
                full_known_reference=full_ref, rows_in_packet=len(rec), arms=rows)


def interval(pairs, cfg):
    sites = sorted({s for s, v in pairs if v is not None})
    values = np.array([np.mean([v for s, v in pairs if s == site and v is not None]) for site in sites])
    n = sum(v is not None for _, v in pairs)
    if not sites:
        return dict(defined_views=0, undefined_views=len(pairs), localities=0, mean=None, CI95=None)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boots = values[rng.integers(0, len(values), (cfg['bootstrap_resamples'], len(values)))].mean(1)
    return dict(defined_views=n, undefined_views=len(pairs)-n, localities=len(sites),
        mean=float(values.mean()), CI95=np.quantile(boots, [.025, .975]).tolist(),
        strict_all_views_mean=float(values.mean()) if n == len(pairs) else None,
        nominal_exposed_development_only=True)


def summarize(groups, cfg):
    arms = {}
    for arm in ARMS:
        rows = [g['arms'][arm] for g in groups]
        d = {}
        for kind in ('parent', 'consensus', 'removed'):
            rr = [r[kind] for r in rows]
            risks = [r['easy_selected_risk_upper'] for r in rr]
            d[kind] = dict(complete_support_pass=sum(r['finite_completion_supported'] for r in rr),
                selected_occurrences=sum(r['selected_count'] for r in rr),
                unknown_occurrences=sum(r['selected_unknown'] for r in rr),
                defined=sum(v is not None for v in risks), undefined=sum(v is None for v in risks),
                upper_violations=sum(v is not None and v > .02+1e-12 for v in risks),
                worst_completion_upper=max((v for v in risks if v is not None), default=None))
        for name, comparator in (('utility_change_vs_parent', 'parent'), ('utility_change_vs_matched_expectation', 'matched')):
            vals = []
            for g in groups:
                r = g['arms'][arm]
                reference = r['parent']['selected_net_gain_lower_mass'] if comparator == 'parent' else r['matched']['conservative_utility_expectation']
                v = 100*(r['consensus']['selected_net_gain_lower_mass']-reference)/g['full_known_reference'] if g['full_known_reference'] > 0 else None
                vals.append((g['source'], v))
            d[name] = interval(vals, cfg)
        d['removed_for_unestimable_deletion'] = sum(r['removed_for_unestimable_deletion'] for r in rows)
        d['removed_despite_estimable_deletions'] = sum(r['removed_despite_estimable_deletions'] for r in rows)
        folds = [f for r in rows for f in r['folds']]
        d['folds'] = len(folds)
        d['estimable_folds'] = sum(f['estimable'] for f in folds)
        d['nested_evaluations'] = sum(f['nested_count'] for f in folds)
        d['nested_changed_evaluations'] = sum(v > 0 for f in folds for v in f['deletion_changed_occurrences'])
        d['changed_occurrences_across_deletions'] = sum(sum(f['deletion_changed_occurrences']) for f in folds)
        d['lost_complete_support'] = sum(r['parent']['finite_completion_supported'] and not r['consensus']['finite_completion_supported'] for r in rows)
        d['gained_complete_support'] = sum(not r['parent']['finite_completion_supported'] and r['consensus']['finite_completion_supported'] for r in rows)
        arms[arm] = d
    return dict(arms=arms, groups=len(groups), source_only=True,
        source_recording_count_distribution=dict(Counter(g['source_recordings'] for g in groups)),
        result_source='fresh_run_nested_source_calibration_on_cached_verified_forecasts',
        inference_uses_outcome_availability=False, new_neural_updates=0, transfer_evaluated=False,
        deployment_promoted=False, independent_roles_read=False, stage5c_executed=False, smc_enabled=False)
