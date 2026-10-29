"""Fixed-roster action-policy readout; matched controls measure utility only."""
import json
from pathlib import Path
import resource
import statistics
import time

import numpy as np

from src.evaluation.m3w_multiaction_support import score_choices
from src.evaluation.m3w_easy_constrained_policy import budget_status


def chosen_values(values, floor, choices):
    v, floor, choices = map(np.asarray, (values, floor, choices))
    if v.shape[:1] != floor.shape[:1] or choices.shape != floor.shape[:1]:
        raise ValueError('Aligned one-action choices required')
    take = choices >= 0
    out = floor.copy()
    out[take] = v[np.flatnonzero(take), choices[take]]
    return out


def paired_bounds(packet, left, right, take=None):
    """Positive means right improves left, including completion worst cases."""
    n = len(left)
    take = np.ones(n, bool) if take is None else np.asarray(take)
    if take.shape != (n,) or take.dtype != bool:
        raise ValueError('Fixed evaluation subset required')
    a = chosen_values(packet['ade'], packet['floor_ade'], left)
    b = chosen_values(packet['ade'], packet['floor_ade'], right)
    known = np.isfinite(a)
    np.testing.assert_array_equal(known, np.isfinite(b))
    pa = chosen_values(packet['rollouts'], packet['floor_rollout'], left)
    pb = chosen_values(packet['rollouts'], packet['floor_rollout'], right)
    envelope = np.linalg.norm(pa-pb, axis=-1).max(1)
    delta = float((a[take & known]-b[take & known]).sum())
    mass = float(envelope[take & ~known].sum())
    reference = float(packet['floor_ade'][take & known].sum())
    return dict(known_delta_mass=delta, unknown_disagreement_mass=mass,
        lower_pp=100*(delta-mass)/reference if reference > 0 else None,
        upper_pp=100*(delta+mass)/reference if reference > 0 else None,
        known_pp=100*delta/reference if reference > 0 else None,
        full_reference_mass_counted_once=reference)


def matched_choices(left, right, left_rank, right_rank, recordings, frames, ids):
    values = [np.asarray(x) for x in (left, right, left_rank, right_rank, recordings, frames, ids)]
    left, right, lr, rr, rec, frame, ids = values
    if any(x.shape != left.shape for x in values):
        raise ValueError('Aligned current-query identities and causal ranks required')
    a, b = np.full_like(left, -1), np.full_like(right, -1)
    groups = {}
    for i, key in enumerate(zip(rec.astype(str), frame.tolist())):
        groups.setdefault(key, []).append(i)
    for rows in groups.values():
        rows = np.asarray(rows)
        yes = [rows[c[rows] >= 0] for c in (left, right)]
        count = min(map(len, yes))
        for output, original, rank, eligible in ((a, left, lr, yes[0]), (b, right, rr, yes[1])):
            order = np.lexsort((ids[eligible], -rank[eligible]))[:count]
            chosen = eligible[order]
            output[chosen] = original[chosen]
    if (a >= 0).sum() != (b >= 0).sum():
        raise AssertionError('Count matching changed intervention counts')
    return a, b


def compact_score(y, envelope, choices):
    score = score_choices(y, envelope, choices)
    return dict(score, budget_status=budget_status(score),
                empty_risk_status='not_estimable' if not np.any(choices >= 0) else 'see_bounds')


def group_readout(packet, choices, ranks, predicted):
    labels, envelope = packet['y'], packet['envelope']
    known = np.isfinite(labels).all((1, 2))
    reference = float(packet['floor_ade'][known].sum())
    policies, records = {}, {}
    for name, choice in choices.items():
        ade = chosen_values(packet['ade'], packet['floor_ade'], choice)
        fde = chosen_values(packet['fde'], packet['floor_fde'], choice)
        oracle = np.minimum(packet['floor_ade'], np.min(np.where(packet['allowed'], packet['ade'], np.inf), 1))
        cv_reference = float(packet['cv_ade'][known].sum())
        fde_known = np.isfinite(packet['floor_fde']); fde_ref = float(packet['floor_fde'][fde_known].sum())
        policies[name] = dict(policy=compact_score(labels, envelope, choice),
            known_ADE_gain_floor_percent=100*(1-ade[known].sum()/reference) if reference > 0 else None,
            known_ADE_gain_CV_percent=100*(1-ade[known].sum()/cv_reference) if cv_reference > 0 else None,
            known_FDE_gain_floor_percent=100*(1-fde[fde_known].sum()/fde_ref) if fde_ref > 0 else None,
            known_regret_over_cost_scale=float(np.mean((ade[known]-oracle[known])/packet['scales'][known])),
            neural_interventions=int((choice == 6).sum()),
            action_counts={str(i):int((choice == i).sum()) for i in range(-1, 7)})
    for recording in sorted(set(packet['recordings'])):
        at = packet['recordings'] == recording
        records[str(recording)] = {name:compact_score(labels[at], envelope[at], choice[at])
                                   for name, choice in choices.items()}
    comparisons = {}
    for other in ('physical_only', 'neural_only', 'frozen_origin_binary'):
        left, right = choices[other], choices['physical_neural']
        aa, bb = matched_choices(left, right, ranks[other], ranks['physical_neural'],
                                 packet['recordings'], packet['frames'], packet['ids'])
        by_record = {}
        for recording in sorted(set(packet['recordings'])):
            at = packet['recordings'] == recording
            by_record[str(recording)] = paired_bounds(packet, left, right, at)
        comparisons['physical_neural_minus_'+other] = dict(full=paired_bounds(packet, left, right),
            matched=paired_bounds(packet, aa, bb), matched_count=int((aa >= 0).sum()),
            matched_empty=not np.any(aa >= 0), matched_role='utility_only_not_safety', by_record=by_record)
    calibration = {}
    for name, prediction in predicted.items():
        mse = np.square((prediction[known]-labels[known])/packet['scales'][known, None, None]).mean((0, 1))
        calibration[name] = dict(normalized_moment_MSE=mse.tolist(),
            unknown_outcomes_imputed=False, evaluation_support='known_rows_all_seven_causal_actions')
    baselines = {}
    for k in range(7):
        error = packet['ade'][known, k]
        baselines[str(k)] = dict(known_ADE_gain_CV_percent=100*(1-error.sum()/cv_reference)
            if cv_reference > 0 else None)
    return dict(rows=len(known), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=reference, policies=policies, recordings=records, comparisons=comparisons,
        calibration=calibration, candidate_accuracy=baselines, inference_label_filter=False)


def interval(rows, getter, cfg):
    grouped = {}
    for row in rows:
        value = getter(row['result'])
        if value is None:
            return dict(point=None, CI95=None, reason='required_view_not_estimable')
        grouped.setdefault(row['source'], []).append(float(value))
    means = {source:statistics.fmean(v) for source, v in sorted(grouped.items())}
    if len(means) != 12:
        raise ValueError('All twelve SOURCE localities required')
    values = np.asarray(list(means.values()))
    bootstrap = np.random.default_rng(cfg['bootstrap_seed']).choice(values,
        size=(cfg['bootstrap_resamples'], len(values)), replace=True).mean(1)
    seeds = {str(seed):statistics.fmean(getter(row['result']) for row in rows if row['seed'] == seed)
             for seed in cfg['seeds']}
    return dict(point=float(values.mean()), CI95=np.quantile(bootstrap, [.025, .975]).tolist(),
        by_locality=means, by_seed=seeds, bootstrap_unit='source_locality_after_context_and_head_seed_average',
        bootstrap_resamples=cfg['bootstrap_resamples'], conditional_exploratory_not_confirmation=True)


def summarize(rows, cfg):
    scopes = {}
    for scope in ('fit', 'inner_held'):
        selected = [r for r in rows if r['scope'] == scope]
        if len(selected) != 72 or len({(r['context'], r['source'], r['seed']) for r in selected}) != 72:
            raise ValueError('Complete72-view scope required')
        names = list(selected[0]['result']['policies'])
        safety, accuracy = {}, {}
        for name in names:
            ps = [r['result']['policies'][name]['policy'] for r in selected]
            records = [record[name] for r in selected for record in r['result']['recordings'].values()]
            safety[name] = dict(required_views=len(ps), budget_views=sum(p['budget_status']['supported'] for p in ps),
                required_record_views=len(records), budget_record_views=sum(p['budget_status']['supported'] for p in records),
                empty_views=sum(p['selected_count'] == 0 for p in ps),
                undefined_easy_record_views=sum(p['easy_selected_risk_upper'] is None for p in records),
                selected=sum(p['selected_count'] for p in ps), selected_unknown=sum(p['selected_unknown'] for p in ps),
                known_all_risk_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk'] > .02+1e-12 for p in records),
                known_easy_risk_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in records))
            accuracy[name] = interval(selected,
                lambda r:r['policies'][name]['known_ADE_gain_floor_percent'], cfg)
        contrasts = {}
        for other in ('physical_only', 'neural_only', 'frozen_origin_binary'):
            key = 'physical_neural_minus_'+other
            contrasts[key] = {mode+'_'+bound:interval(selected,
                lambda r, m=mode, b=bound:r['comparisons'][key][m][b], cfg)
                for mode in ('full', 'matched') for bound in ('lower_pp', 'upper_pp', 'known_pp')}
            contrasts[key]['empty_matched_views'] = sum(r['result']['comparisons'][key]['matched_empty'] for r in selected)
        scopes[scope] = dict(views=len(selected), safety=safety, accuracy=accuracy, contrasts=contrasts,
            unique_source_recordings=len({(r['source'], key) for r in selected for key in r['result']['recordings']}))
    primary = scopes['inner_held']['safety']['physical_neural']
    effect = scopes['inner_held']['contrasts']['physical_neural_minus_physical_only']['full_lower_pp']
    full_budget = primary['budget_views'] == primary['required_views'] and primary['budget_record_views'] == primary['required_record_views']
    return dict(result_source='fresh_run_fixed_final_SOURCE_TRAIN_internal_readout', scopes=scopes,
        primary='physical_neural', primary_comparator='physical_only',
        primary_full_roster_budget_supported=full_budget,
        neural_increment_conditional_CI_positive=effect['CI95'] is not None and effect['CI95'][0] > 0,
        independent_roles_opened=False, independent_confirmation=False, whole_pipeline_OOF=False,
        matched_controls_utility_only=True, empty_risk_is_not_estimable=True,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def evaluate(run, *, verify=False, resume=False):
    cfg, reg, _, data, jobs, identity = run.admission()
    learned = run.models(); finish = run.PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed scientific readout immutable; use verify')
    started, beat, guard = run.runtime(cfg)
    _, origin_refs = run.old.direct.ORIGINS.models(reg)
    rows, refs = [], []
    for context, items in sorted(reg['contexts'].items()):
        scopes = run.prepare_context(context, items, data, jobs, identity, beat, include_held=True)
        for seed in cfg['seeds']:
            states = {arm:run.api.read_checkpoint(run.ROOT/learned[context, seed, arm]['checkpoint']['path']) for arm in run.api.ARMS}
            origin_ref = origin_refs[context, seed, 'self']['checkpoint']
            origin = run.old.base.core.read_checkpoint(run.ROOT/origin_ref['path'])
            for scope, packets in scopes.items():
                originals = run.old.direct.origin_predictions(origin, {source:p['args'] for source,p in packets.items()})
                for source, packet in sorted(packets.items()):
                    path = run.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit resume required for partial readout')
                    choices, ranks, predictions = {}, {}, {}
                    for arm, state in states.items():
                        predictions[arm] = run.api.predict(state, packet['x'], packet['envelope'], packet['scales'])
                        for multiplier in (1., *cfg['sensitivity_harm_multipliers']):
                            name = arm if multiplier == 1 else arm+'_harm'+str(int(multiplier))
                            choices[name], ranks[name] = run.api.select_actions(predictions[arm], packet['allowed'],
                                arm=arm, harm_multiplier=multiplier)
                    p = originals[source]
                    historical = run.old.base.actions(p, run.old.base.shared.eligible(packet['args']))
                    causal_real = historical & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(causal_real, 6, -1)
                    ranks['frozen_origin_binary'] = p[:, 0]-p[:, 1]
                    choices['floor'] = np.full(len(p), -1, np.int64); ranks['floor'] = np.zeros(len(p))
                    result = group_readout(packet, choices, ranks, predictions)
                    result['old_origin_analytic_noops_removed'] = int((historical & ~causal_real).sum())
                    value = dict(context=context, source=source, seed=seed, scope=scope, result=result,
                        action_sha256={name:run.fingerprint(v) for name,v in choices.items()},
                        prediction_sha256={name:run.fingerprint(v) for name,v in predictions.items()},
                        registration_sha256=run.sha(run.REG), training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
                        original_binary_checkpoint_sha256=origin_ref['sha256'],
                        result_source='fresh_run_source_internal_cached_verified_forecaster_controls', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact fixed action/cost readout replay differs')
                    else:
                        run.once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(run.ROOT)), sha256=run.sha(path)))
            beat(state='readout_context_complete', context=context, views=len(rows), total=144)
        del scopes
    summary = summarize(rows, cfg)
    if verify:
        old = json.loads(finish.read_text())
        if old['groups'] != refs or summary != json.loads((run.PUBLIC/'summary.json').read_text()):
            raise ValueError('Exact summary replay differs')
        run.once(run.PUBLIC/'verification.json', dict(exact_replay=True, summary_sha256=run.sha(run.PUBLIC/'summary.json'),
            groups=len(refs), registration_sha256=run.sha(run.REG)))
    else:
        run.once(run.PUBLIC/'summary.json', summary)
        run.once(finish, dict(groups=refs, summary_sha256=run.sha(run.PUBLIC/'summary.json'),
            registration_sha256=run.sha(run.REG), training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
            pid=run.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_opened=False))
    run.json_write(run.PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(), seconds=time.monotonic()-started,
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, exact_replay=verify))
    beat(state='readout_exact_verified' if verify else 'readout_complete',
         primary_full_roster_budget_supported=summary['primary_full_roster_budget_supported'],
         neural_increment_conditional_CI_positive=summary['neural_increment_conditional_CI_positive'])
