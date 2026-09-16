"""Development-only matched intervention evaluation; no confirmation or certification.

All decisions precede label access. Primary pooled comparisons require explicitly
approved past-normalized errors; unverified dataset-local errors stay per recording.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_unification.m3w_causal_recordings import restore_scene_rollouts
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_supervised_intervention import (
    pack_inputs, collate_inputs, risk_features, predict_linear_gain_harm, load_verified_forecaster,
)
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions,
    compare_at_independent_coverage,
)
import torch


ARMS = ('floor', 'uncontrolled', 'independent', 'scene_uniform', 'joint')
POLICY_FIELDS = {'pair_weight', 'max_mean_predicted_harm', 'max_intervention_fraction',
                 'min_predicted_gain', 'max_agent_predicted_harm'}


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scene_requests(reader, task, *, stride):
    """Enumerate observable timestamps, not future-complete target-index rows.

    Different native past grids are evaluated separately, never interpolated into
    a fictitious synchronous forecast. History-only unsupported agents are logged.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError('Explicit positive scene-query stride required')
    k = task['history_steps']
    for frame in np.unique(reader.frame_values)[::stride]:
        if task['prediction_unit'] == 'raw_annotation_frames':
            horizons = {task['horizon']}
        elif task['prediction_unit'] == 'observation_steps':
            horizons = set()
            lo, hi = np.searchsorted(reader.frame_values, frame, side='left'), np.searchsorted(reader.frame_values, frame, side='right')
            for current in reader.frame_order[lo:hi]:
                begin = int(current) - k + 1
                if begin < reader._track_start(int(current)):
                    continue
                deltas = np.diff(reader.points[begin:int(current) + 1, 0])
                if deltas[-1] > 0 and np.all(deltas == deltas[-1]):
                    horizons.add(int(deltas[-1]) * task['horizon'])
        else:
            raise ValueError('Explicit supported request time unit required')
        for horizon in sorted(horizons):
            scene = reader.get_scene_inputs(int(frame), horizon, history_steps=k)
            if task['prediction_unit'] == 'observation_steps':
                keep, dropped = [], []
                for agent in scene['agents']:
                    if len(agent['inputs']['prediction_frame_offsets']) == task['horizon']:
                        keep.append(agent)
                    else:
                        dropped.append({'agent_id': agent['agent_id'], 'reason': 'different_native_request_grid'})
                scene['agents'] = keep
                scene['excluded_past_support'].extend(dropped)
            grids = {}
            for agent in scene['agents']:
                grid = tuple(agent['inputs']['prediction_frame_offsets'].tolist())
                grids.setdefault(grid, []).append(agent)
            for grid, agents in sorted(grids.items()):
                excluded = [a for a in scene['agents'] if tuple(a['inputs']['prediction_frame_offsets'].tolist()) != grid]
                yield {**scene, 'agents': agents, 'excluded_past_support': [*scene['excluded_past_support'],
                       *({'agent_id': a['agent_id'], 'reason': 'different_native_request_grid'} for a in excluded)]}


def _validate_policy(policy):
    if (set(policy) != POLICY_FIELDS or not np.isfinite(list(policy.values())).all()
            or min(policy.values()) < 0 or policy['max_intervention_fraction'] > 1):
        raise ValueError('Explicit finite policy weights, support rule and budgets required')


def decide_scene(scene, forecaster, head, *, baseline, policy, geometry, device, solver_seconds,
                 include_matched_coverage=False):
    """Pure inference boundary: no reader, label array or future-validity mask."""
    _validate_policy(policy)
    if not isinstance(include_matched_coverage, bool):
        raise ValueError('Matched-coverage diagnostic requires an explicit Boolean option')
    agents = scene['agents']
    inputs = collate_inputs([pack_inputs(a['inputs'], baseline) for a in agents])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    forecaster.eval()
    with torch.no_grad():
        candidate = forecaster(inputs)
        if candidate.shape != inputs['baseline'].shape:
            raise ValueError('Candidate forecast schema mismatch')
        finite_forecast = torch.isfinite(candidate).all(dim=(1, 2))
        candidate = torch.where(finite_forecast[:, None, None], candidate, inputs['baseline'])
        scores = predict_linear_gain_harm(head, risk_features(inputs, candidate).cpu().numpy())
    b, c = inputs['baseline'].cpu().numpy(), candidate.cpu().numpy()
    ids = [a['agent_id'] for a in agents]
    common_b = restore_scene_rollouts(scene, dict(zip(ids, b)))['xy_dataset_local']
    common_c = restore_scene_rollouts(scene, dict(zip(ids, c)))['xy_dataset_local']
    current = np.stack([a['coordinate_transform']['origin_xy'] for a in agents])
    edges = past_proximity_edges(current, radius=geometry['graph_radius'])
    pairs = proximity_cost_table(common_b, common_c, edges, distance_threshold=geometry['proximity_threshold'])
    finite_scores = np.isfinite(scores['gain']) & np.isfinite(scores['harm'])
    available = finite_forecast.cpu().numpy() & finite_scores
    scores = {k: np.where(available, v, 0.) for k, v in scores.items()}
    supported = available & (scores['gain'] >= policy['min_predicted_gain']) & (scores['harm'] <= policy['max_agent_predicted_harm'])
    problem = InterventionProblem(scores['gain'], scores['harm'], supported, edges, pairs,
                                  policy['pair_weight'], policy['max_mean_predicted_harm'],
                                  int(np.floor(len(agents) * policy['max_intervention_fraction'])))
    arms = {}
    for mode in ARMS:
        # The uncontrolled comparator uses every finite candidate, without score gating.
        if mode == 'uncontrolled':
            original = problem.supported
            problem.supported = available.copy()
            decision = select_interventions(problem, mode=mode, time_limit_seconds=solver_seconds)
            problem.supported = original
        else:
            decision = select_interventions(problem, mode=mode, time_limit_seconds=solver_seconds)
        arms[mode] = {**decision, 'prediction': np.where(decision['switch'][:, None, None], c, b)}
    result = {'agent_ids': ids, 'baseline': b, 'candidate': c, 'arms': arms,
            'predicted_gain': scores['gain'], 'predicted_harm': scores['harm'],
            'supported': supported, 'common_coordinate_graph_edges': len(edges),
            'nonfinite_candidate_or_scores_floor_count': int((~available).sum()),
            'matching': 'same_predictor_and_budget_caps_not_equal_realized_coverage',
            'calibrated_risk': False}
    if include_matched_coverage:
        comparison = compare_at_independent_coverage(problem, time_limit_seconds=solver_seconds)
        for name in ('reference', 'joint_exact'):
            choice = comparison[name]
            choice['prediction'] = np.where(choice['switch'][:, None, None], c, b)
        result['coverage_match'] = comparison
    return result


def score_scene(scene, decisions, labels, *, label_policy):
    if label_policy not in {'complete_requested_path', 'available_steps'}:
        raise ValueError('Explicit label coverage policy required')
    agents = scene['agents']
    if decisions['agent_ids'] != [a['agent_id'] for a in agents] or len(labels) != len(agents):
        raise ValueError('Prediction/label agent identity mismatch')
    rows = []
    for i, (agent, label) in enumerate(zip(agents, labels)):
        if label['agent_id'] != agent['agent_id']:
            raise ValueError('Label agent identity mismatch')
        request = scene['frame_id'] + agent['inputs']['prediction_frame_offsets']
        if not np.array_equal(label['future_frame_ids'], request):
            raise ValueError('Label grid differs from requested timestamps')
        t = agent['coordinate_transform']
        y = (label['future_xy_dataset_local'] - t['origin_xy']) @ t['rotation'] / t['scale']
        mask = np.asarray(label['future_label_mask'])
        if mask.dtype != bool or mask.shape != (len(request),) or not np.isfinite(y[mask]).all():
            raise ValueError('Invalid target label mask or valid coordinates')

        def errors(prediction):
            error = np.linalg.norm(prediction - y, axis=-1)
            ade_valid = mask.all() if label_policy == 'complete_requested_path' else mask.any()
            return {'ade': float(error[mask].mean()) if ade_valid else None,
                    'fde': float(error[-1]) if mask[-1] else None}

        b = errors(decisions['baseline'][i])
        row = {k: scene[k] for k in ('recording_id', 'physical_scene', 'frame_id', 'horizon_raw')}
        row.update(agent_id=agent['agent_id'], scale=float(t['scale']), baseline_ade=b['ade'], baseline_fde=b['fde'],
                   requested_steps=len(request), available_label_steps=int(mask.sum()), arms={})
        for arm, choice in decisions['arms'].items():
            row['arms'][arm] = {**errors(choice['prediction'][i]), 'switch': bool(choice['switch'][i]),
                                'pair_proxy': choice['mean_pair_proxy'], 'reason': choice['reason'],
                                'constraints_satisfied': choice['predicted_constraints_satisfied']}
        rows.append(row)
    return rows


def _cluster_contributions(rows, values, aggregation):
    """Columns share the same eligible agents; each resampled item is a scene."""
    scenes = sorted({r['physical_scene'] for r in rows})
    numerator, denominator = [], []
    for scene in scenes:
        ids = [i for i, r in enumerate(rows) if r['physical_scene'] == scene]
        v = values[ids]
        if aggregation == 'agent_window':
            numerator.append(v.sum(0)); denominator.append(len(v))
        elif aggregation == 'equal_physical_scene':
            numerator.append(v.mean(0)); denominator.append(1)
        elif aggregation == 'equal_recording':
            recordings = sorted({rows[i]['recording_id'] for i in ids})
            means = [values[[i for i in ids if rows[i]['recording_id'] == recording]].mean(0) for recording in recordings]
            numerator.append(np.sum(means, axis=0)); denominator.append(len(means))
        else:
            raise ValueError('Unknown approved aggregation')
    return scenes, np.asarray(numerator), np.asarray(denominator)


def _slice_metrics(rows, arm, *, metric, aggregation, n_bootstrap, seed):
    eligible = [r for r in rows if r[f'baseline_{metric}'] is not None]
    if not eligible:
        return {'status': 'not_run_no_eligible_labels', 'count': 0}
    values = np.array([[r[f'baseline_{metric}'], r['arms'][arm][metric],
                       max(r['arms'][arm][metric] - r[f'baseline_{metric}'], 0.),
                       float(r['arms'][arm]['switch'])] for r in eligible])
    scenes, sums, counts = _cluster_contributions(eligible, values, aggregation)
    mean = sums.sum(0) / counts.sum()
    improvement = 100 * (mean[0] - mean[1]) / mean[0] if mean[0] > 0 else None
    ci = {'status': 'not_run_insufficient_physical_scenes', 'physical_scene_count': len(scenes),
          'resampling_unit': 'physical_scene', 'independence_verified': False}
    if len(scenes) >= 2:
        rng = np.random.default_rng(seed)
        draw = rng.integers(len(scenes), size=(n_bootstrap, len(scenes)))
        bootstrap = sums[draw].sum(1) / counts[draw].sum(1)[:, None]
        excess = bootstrap[:, 1] - bootstrap[:, 0]
        ratios = 100 * (bootstrap[:, 0] - bootstrap[:, 1]) / np.where(bootstrap[:, 0] > 0, bootstrap[:, 0], np.nan)
        ci.update(status='computed_development_descriptive_not_confirmation', resamples=n_bootstrap,
                  excess_mean_ci95=np.quantile(excess, [.025, .975]).tolist(),
                  improvement_pct_ci95=np.quantile(ratios, [.025, .975]).tolist() if np.isfinite(ratios).all() else None,
                  undefined_ratio_resamples=int((~np.isfinite(ratios)).sum()))
    scene_means = [values[[i for i, r in enumerate(eligible) if r['physical_scene'] == s], 1].mean() for s in scenes]
    return {'status': 'evaluated', 'count': len(eligible), 'baseline_error': float(mean[0]),
            'selected_error': float(mean[1]), 'mean_excess_over_floor': float(mean[1]-mean[0]),
            'mean_positive_harm': float(mean[2]), 'switch_rate_valid_labels': float(mean[3]),
            'improvement_pct': improvement, 'degradation_fraction': -improvement/100 if improvement is not None else None,
            'agent_window_p95_error_descriptive': float(np.quantile(values[:, 1], .95)),
            'worst_physical_scene_mean_error': float(max(scene_means)), 'bootstrap': ci}


def summarize_rows(rows, *, metric, aggregation, error_unit, easy_threshold, hard_threshold,
                   bootstrap_resamples, bootstrap_seed):
    if error_unit != 'past_normalized':
        raise ValueError('Only explicitly approved past-normalized pooled selection is implemented; raw errors stay per recording')
    if metric not in {'ade', 'fde'} or aggregation not in {'equal_physical_scene', 'equal_recording', 'agent_window'}:
        raise ValueError('Explicit primary metric and aggregation required')
    if not rows or not 0 <= easy_threshold < hard_threshold or bootstrap_resamples < 2000:
        raise ValueError('Nonempty rows, declared slice thresholds and at least 2000 cluster resamples required')
    identities = [(r['recording_id'], r['agent_id'], r['frame_id'], r['horizon_raw']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate evaluated agent query')
    arms = set(rows[0]['arms'])
    if any(set(r['arms']) != arms for r in rows):
        raise ValueError('All control arms must evaluate the same agent queries')
    for row in rows:
        for m in ('ade', 'fde'):
            baseline = row[f'baseline_{m}']
            for arm in arms:
                value = row['arms'][arm][m]
                if (baseline is None) != (value is None) or (value is not None and (not np.isfinite([baseline, value]).all() or min(baseline, value) < 0)):
                    raise ValueError('Arms differ in label eligibility or have invalid error')
    fixed = {'metric': metric, 'aggregation': aggregation, 'n_bootstrap': bootstrap_resamples, 'seed': bootstrap_seed}
    slices = {'all': rows,
              'easy': [r for r in rows if r[f'baseline_{metric}'] is not None and r[f'baseline_{metric}'] <= easy_threshold],
              'hard': [r for r in rows if r[f'baseline_{metric}'] is not None and r[f'baseline_{metric}'] >= hard_threshold]}
    result = {'primary_metric': metric, 'aggregation': aggregation, 'error_unit': error_unit,
              'physical_scene_count': len({r['physical_scene'] for r in rows}),
              'recording_count': len({r['recording_id'] for r in rows}), 'agent_query_count': len(rows),
              'unique_scene_queries': len({(r['recording_id'], r['frame_id'], r['horizon_raw'], r['requested_steps']) for r in rows}),
              'label_coverage': {m: sum(r[f'baseline_{m}'] is not None for r in rows) for m in ('ade', 'fde')},
              'comparison_identity': content_digest(sorted([(key, r['scale'], r['baseline_ade'], r['baseline_fde']) for key, r in zip(identities, rows)])),
              'slice_definition': {'easy_baseline_error_at_most': easy_threshold, 'hard_baseline_error_at_least': hard_threshold},
              'arms': {}}
    for arm in sorted(arms):
        result['arms'][arm] = {s: _slice_metrics(part, arm, **fixed) for s, part in slices.items()}
        result['arms'][arm]['per_physical_scene'] = {
            scene: _slice_metrics([r for r in rows if r['physical_scene'] == scene], arm, **fixed)
            for scene in sorted({r['physical_scene'] for r in rows})}
        result['arms'][arm]['per_raw_horizon'] = {
            str(h): _slice_metrics([r for r in rows if r['horizon_raw'] == h], arm, **fixed)
            for h in sorted({r['horizon_raw'] for r in rows})}
        other_metric = 'fde' if metric == 'ade' else 'ade'
        result['arms'][arm]['secondary_' + other_metric] = _slice_metrics(rows, arm, **{**fixed, 'metric': other_metric})
        pair_by_query = {(r['recording_id'], r['frame_id'], r['horizon_raw'], r['requested_steps']): r['arms'][arm]['pair_proxy'] for r in rows}
        result['arms'][arm].update(switch_rate_all_past_supported=float(np.mean([r['arms'][arm]['switch'] for r in rows])),
                                  mean_query_excess_proximity_proxy=float(np.mean(list(pair_by_query.values()))),
                                  decisions=Counter(r['arms'][arm]['reason'] for r in rows),
                                  budget_violation_agent_queries=sum(not r['arms'][arm]['constraints_satisfied'] for r in rows))
    raw = {}
    for recording in sorted({r['recording_id'] for r in rows}):
        part = [r for r in rows if r['recording_id'] == recording]
        raw[recording] = {'coordinate_claim': 'dataset_local_unverified', 'arms': {}, 'baseline': {}}
        for m in ('ade', 'fde'):
            eligible = [r for r in part if r[f'baseline_{m}'] is not None]
            raw[recording]['baseline'][m] = float(np.mean([r[f'baseline_{m}'] * r['scale'] for r in eligible])) if eligible else None
            for arm in sorted(arms):
                raw[recording]['arms'].setdefault(arm, {})[m] = float(np.mean([r['arms'][arm][m] * r['scale'] for r in eligible])) if eligible else None
    result['dataset_local_metrics'] = {'status': 'per_recording_only_no_cross_domain_raw_pool', 'recordings': raw}
    result['matched_comparison'] = 'identical_forecasts_queries_and_budget_caps; actual_coverage_reported_not_assumed_equal'
    result['confirmation_result'] = False
    return result


def choose_development(summaries, *, eligible_arms, easy_degradation_max):
    if not summaries or not eligible_arms or not set(eligible_arms) <= set(ARMS) - {'floor', 'uncontrolled'}:
        raise ValueError('Explicit nonempty guarded development family required')
    keys = ('comparison_identity', 'primary_metric', 'aggregation', 'error_unit', 'slice_definition')
    reference = next(iter(summaries.values()))
    if any(any(s[k] != reference[k] for k in keys) for s in summaries.values()):
        raise ValueError('Development selection requires matched sample identity, baseline and aggregation')
    choices, rejections = [], []
    for candidate, summary in sorted(summaries.items()):
        for arm in eligible_arms:
            all_metrics, easy = summary['arms'][arm]['all'], summary['arms'][arm]['easy']
            reasons = []
            if all_metrics['status'] != 'evaluated' or all_metrics.get('mean_excess_over_floor', 0) >= 0:
                reasons.append('no_positive_primary_gain')
            if easy['status'] != 'evaluated':
                reasons.append('no_easy_support')
            elif easy['degradation_fraction'] is None:
                if easy['selected_error'] > easy['baseline_error']:
                    reasons.append('easy_degradation_zero_denominator')
            elif easy['degradation_fraction'] > easy_degradation_max:
                reasons.append('easy_degradation')
            if summary['arms'][arm]['budget_violation_agent_queries']:
                reasons.append('predicted_budget_violation')
            if reasons:
                rejections.append({'candidate_id': candidate, 'arm': arm, 'reasons': reasons})
            else:
                choices.append((all_metrics['selected_error'], summary['arms'][arm]['switch_rate_all_past_supported'], candidate, arm))
    best = min(choices) if choices else None
    return {'selected': {'candidate_id': best[2] if best else None, 'arm': best[3] if best else 'floor'},
            'rejections': rejections, 'selection_role': 'development_only',
            'selection_rule': 'minimum_primary_error_subject_to_easy_and_predicted_budget_guards; floor_if_none',
            'tie_break': 'lower_actual_intervention_then_lexicographic_identity',
            'calibrated_risk': False, 'deployable': False, 'confirmation_evaluated': False}


def validate_plan(contract, plan):
    contract._assert_frozen()
    rules = contract.protocol.get('development_evaluation')
    if not rules:
        raise ValueError('Explicit approved development_evaluation rules required')
    required = {'error_unit', 'label_policy', 'query_stride', 'geometry_by_recording', 'easy_threshold', 'hard_threshold',
                'eligible_arms', 'solver_seconds', 'bootstrap_seed', 'policies'}
    if set(rules) != required or rules['error_unit'] != 'past_normalized':
        raise ValueError('Complete explicit normalized development evaluation rules required')
    if rules['label_policy'] not in {'complete_requested_path', 'available_steps'} or type(rules['query_stride']) is not int or rules['query_stride'] < 1:
        raise ValueError('Explicit valid query and label-coverage rules required')
    if not np.isfinite([rules['easy_threshold'], rules['hard_threshold'], rules['solver_seconds']]).all() or not 0 <= rules['easy_threshold'] < rules['hard_threshold'] or rules['solver_seconds'] <= 0:
        raise ValueError('Explicit finite slice thresholds and solver limit required')
    if not rules['eligible_arms'] or not set(rules['eligible_arms']) <= {'independent', 'scene_uniform', 'joint'}:
        raise ValueError('Only declared guarded arms may be selected')
    if type(rules['bootstrap_seed']) is not int:
        raise ValueError('Explicit bootstrap RNG seed required')
    expected_easy = {'kind': 'baseline_error_at_most', 'metric': contract.protocol['task']['primary_metric'],
                     'error_unit': rules['error_unit'], 'threshold': rules['easy_threshold']}
    if contract.protocol['risk']['easy_definition'] != expected_easy:
        raise ValueError('Development easy slice differs from the approved risk definition')
    recordings = sorted(r for r, role in contract.protocol['assignments'].items() if role == 'development')
    if set(rules['geometry_by_recording']) != set(recordings):
        raise ValueError('Dataset-local geometry must be declared per development recording')
    for geometry in rules['geometry_by_recording'].values():
        if set(geometry) != {'graph_radius', 'proximity_threshold'} or not np.isfinite(list(geometry.values())).all() or min(geometry.values()) <= 0:
            raise ValueError('Finite positive dataset-local proxy geometry required')
    for policy in rules['policies'].values():
        _validate_policy(policy)
    if plan.get('schema_version') != 1 or not plan.get('candidates'):
        raise ValueError('Frozen nonempty development candidate family required')
    ids = [c['id'] for c in plan['candidates']]
    if len(set(ids)) != len(ids) or any(not isinstance(n, str) or not n for n in ids):
        raise ValueError('Unique nonempty candidate identities required')
    if len({c['baseline'] for c in plan['candidates']}) != 1:
        raise ValueError('Matched comparison requires the same explicit baseline floor')
    for c in plan['candidates']:
        if c['policy_id'] not in rules['policies']:
            raise ValueError('Policy outside approved candidate grid')
        for field in ('forecaster_id', 'risk_head_id'):
            contract.assert_prediction_use(c[field], recordings, purpose='development')
        path = contract._path(c['risk_report_path'])
        if file_digest(path) != c['risk_report_sha256']:
            raise ValueError('Cost-head report identity changed')
    return rules, recordings


def _load_cost_head(contract, candidate):
    artifact = contract.artifacts[candidate['risk_head_id']]
    if artifact['kind'] != 'risk_head':
        raise ValueError('Expected risk-head artifact')
    report = json.loads(contract._path(candidate['risk_report_path']).read_text())
    if (report['checkpoint_sha256'] != artifact['sha256'] or report['protocol_sha256'] != contract.digest
            or sorted(report['fit_recordings']) != sorted(artifact['fit_recordings'])
            or sorted(report['parents']) != sorted(artifact['parents'])
            or report['normalization_source'] != 'fit_OOF_rows_only'
            or report['baseline_name'] != candidate['baseline']
            or report['metric'] != contract.protocol['task']['primary_metric']
            or report['code_sha256'] != file_digest(Path(__file__).parents[1] / 'world_model/m3w_supervised_intervention.py')):
        raise ValueError('Cost-head supervision/lineage mismatch')
    with np.load(contract._path(artifact['path']), allow_pickle=False) as arrays:
        head = {k: arrays[k].copy() for k in ('mean', 'scale', 'coef', 'intercept')}
    if any(not np.isfinite(v).all() for v in head.values()) or np.any(head['scale'] <= 0):
        raise ValueError('Invalid fitted cost-head normalization or weights')
    return head


def evaluate_development(contract, plan, *, device='cpu', on_recording=None, cached_rows=None, progress=None):
    rules, recordings = validate_plan(contract, plan)
    summaries, all_rows = {}, {}
    for candidate in plan['candidates']:
        model = load_verified_forecaster(contract, candidate['forecaster_id'], device=device)
        if model._fitted_baseline_name != candidate['baseline']:
            raise ValueError('Forecast baseline feature identity mismatch')
        head = _load_cost_head(contract, candidate)
        rows = []
        for recording in recordings:
            reader, _ = contract.open_recording(recording, purpose='development')
            key = (candidate['id'], recording)
            part = cached_rows.get(key) if cached_rows else None
            if part is None:
                part = []
                for scene in scene_requests(reader, contract.protocol['task'], stride=rules['query_stride']):
                    decision = decide_scene(scene, model, head, baseline=candidate['baseline'],
                                            policy=rules['policies'][candidate['policy_id']],
                                            geometry=rules['geometry_by_recording'][recording],
                                            device=device, solver_seconds=rules['solver_seconds'])
                    labels = reader.get_scene_labels(scene)
                    part.extend(score_scene(scene, decision, labels, label_policy=rules['label_policy']))
                    if progress:
                        progress(key, scene['frame_id'], len(part))
                if on_recording:
                    on_recording(key, part)
            if any(r['recording_id'] != recording or r['physical_scene'] != reader.metadata['physical_scene'] for r in part):
                raise ValueError('Cached evaluation row belongs to a different recording/scene')
            rows.extend(part)
        summaries[candidate['id']] = summarize_rows(rows, metric=contract.protocol['task']['primary_metric'],
                aggregation=contract.protocol['task']['aggregation'], error_unit=rules['error_unit'],
                easy_threshold=rules['easy_threshold'], hard_threshold=rules['hard_threshold'],
                bootstrap_resamples=contract.protocol['bootstrap_resamples'], bootstrap_seed=rules['bootstrap_seed'])
        all_rows[candidate['id']] = rows
    selection = choose_development(summaries, eligible_arms=rules['eligible_arms'],
                                   easy_degradation_max=contract.protocol['risk']['easy_degradation_max'])
    return {'summaries': summaries, 'selection': selection, 'evaluation_role': 'development',
            'protocol_sha256': contract.digest, 'plan_sha256': content_digest(plan),
            'calibration_or_confirmation_opened': False, 'strongest_floor_claim': False}, all_rows
