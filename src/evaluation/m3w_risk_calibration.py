"""Frozen-policy scene-level risk screening, not an unconditional safety claim.

Calibration labels cannot refit predictors, costs, normalizers or thresholds.
Missing labels on an intervention receive the declared worst bounded loss.
The unchanged baseline has zero excess loss, not zero forecasting error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation.m3w_experiment_contract import file_digest, _check_claim, _check_claim_path
from src.evaluation.m3w_development_evaluation import (
    scene_requests, decide_scene, score_scene, _load_cost_head, content_digest,
)
from src.world_model.m3w_joint_intervention import screen_cluster_risks
from src.world_model.m3w_supervised_intervention import load_verified_forecaster


def validate_rules(contract):
    contract._assert_frozen()
    rules = contract.protocol.get('calibration_evaluation', {})
    required = {'error_unit', 'label_policy', 'query_stride', 'geometry_by_recording',
                'solver_seconds', 'within_scene_aggregation', 'selection_rule', 'policy_priority'}
    if set(rules) != required or rules['error_unit'] != 'past_normalized':
        raise ValueError('Explicit approved calibration_evaluation rules required')
    if rules['label_policy'] != 'complete_requested_path':
        raise ValueError('Calibration requires complete requested ADE or exact endpoint FDE; no partial-path risk substitution')
    if (type(rules['query_stride']) is not int or rules['query_stride'] < 1
            or not np.isfinite(rules['solver_seconds']) or rules['solver_seconds'] <= 0):
        raise ValueError('Explicit finite query stride/solver limit required')
    if rules['within_scene_aggregation'] not in {'agent_window', 'equal_recording'}:
        raise ValueError('Explicit within-scene aggregation required')
    if rules['selection_rule'] != 'first_accepted_in_frozen_development_order':
        raise ValueError('Calibration cannot optimize policy order using its outcomes')
    priority = rules['policy_priority']
    if (not isinstance(priority, list) or not priority or any(not isinstance(p, str) or not p for p in priority)
            or len(set(priority)) != len(priority)):
        raise ValueError('Explicit unique policy priority must be frozen in the protocol')
    recordings = sorted(r for r, role in contract.protocol['assignments'].items() if role == 'calibration')
    if set(rules['geometry_by_recording']) != set(recordings):
        raise ValueError('Explicit calibration-recording geometry required')
    for geometry in rules['geometry_by_recording'].values():
        if (set(geometry) != {'graph_radius', 'proximity_threshold'}
                or not np.isfinite(list(geometry.values())).all() or min(geometry.values()) <= 0):
            raise ValueError('Invalid dataset-local graph/proximity settings')
    risks = contract.protocol['risk']['risks']
    if len({r['name'] for r in risks}) != len(risks):
        raise ValueError('Duplicate bounded-risk names')
    for risk in risks:
        fields = {'name', 'statistic', 'lower', 'upper', 'tolerance'}
        statistic = risk.get('statistic')
        if statistic == 'clipped_positive_excess':
            fields.add('clip_scale')
            if not np.isfinite(risk.get('clip_scale', np.nan)) or risk['clip_scale'] <= 0:
                raise ValueError('Positive explicit clipping scale required')
        elif statistic == 'harm_event':
            fields.add('margin')
            if not np.isfinite(risk.get('margin', np.nan)) or risk['margin'] < 0:
                raise ValueError('Nonnegative explicit event margin required')
        else:
            raise ValueError('Unsupported risk functional; a risk name alone is not a definition')
        if set(risk) != fields or risk['lower'] != 0 or risk['upper'] != 1:
            raise ValueError('Supported risk functions must have explicit [0, 1] bounds')
    return rules, recordings


def load_frozen_policy(contract, artifact_id, *, device):
    """Consume the actual development export and its fixed implementation receipt."""
    _, recordings = validate_rules(contract)
    contract.assert_prediction_use(artifact_id, recordings, purpose='calibration')
    artifact = contract.artifacts[artifact_id]
    if artifact['kind'] != 'policy' or artifact['fit_recordings'] or artifact['calibration_recordings']:
        raise ValueError('Expected an unfitted development-selected policy artifact')
    development = sorted(r for r, role in contract.protocol['assignments'].items() if role == 'development')
    if sorted(artifact['selection_recordings']) != development:
        raise ValueError('Policy must declare its full development exposure')
    path = contract._path(artifact['path'])
    body = json.loads(path.read_text())
    if body['protocol_sha256'] != contract.digest or body['baseline_name'] not in BASELINES:
        raise ValueError('Policy protocol/baseline mismatch')
    if body.get('deployment_approved') is not False or body.get('calibrated_risk') is not False:
        raise ValueError('Calibration needs an uncalibrated development export')
    directory = path.parent
    report_path, identity_path = directory / 'development_report.json', directory / 'run_identity.json'
    complete = json.loads((directory / 'completion.json').read_text())
    identity = json.loads(identity_path.read_text())
    if complete['run_sha256'] != content_digest(identity):
        raise ValueError('Development run identity changed')
    root = Path(__file__).resolve().parents[2]
    dependencies = ('src/evaluation/m3w_development_evaluation.py', 'src/world_model/m3w_supervised_intervention.py',
                    'src/world_model/m3w_joint_intervention.py', 'scripts/evaluate_m3w_development.py')
    if any(identity.get('code_sha256', {}).get(name) != file_digest(root / name) for name in dependencies):
        raise ValueError('Development policy implementation changed; re-evaluate development before calibration')
    for name, digest in complete['artifacts'].items():
        item = (directory / name).resolve()
        if not item.is_relative_to(directory.resolve()) or file_digest(item) != digest:
            raise ValueError('Development completion artifact changed')
    if (complete['artifacts'].get(path.name) != artifact['sha256']
            or complete['artifacts'].get(report_path.name) != body['development_report_sha256']
            or file_digest(report_path) != body['development_report_sha256']):
        raise ValueError('Selected policy/report is not bound to development completion')
    report = json.loads(report_path.read_text())
    if (report['protocol_sha256'] != contract.digest or identity['protocol_sha256'] != contract.digest
            or report['evaluation_role'] != 'development' or report['selection']['selected'] != body['selection']
            or report['plan_sha256'] != content_digest(identity['plan'])):
        raise ValueError('Policy differs from its development selection')
    arm, candidate = body['selection']['arm'], body['candidate']
    if candidate is None:
        if arm != 'floor' or body['selection']['candidate_id'] is not None or artifact['parents']:
            raise ValueError('Invalid analytical baseline-only policy')
        return {'id': artifact_id, 'arm': arm, 'baseline': body['baseline_name'], 'model': None, 'head': None}
    if arm not in contract.protocol['development_evaluation']['eligible_arms']:
        raise ValueError('Unapproved guarded policy arm')
    expected = next((c for c in identity['plan']['candidates'] if c['id'] == body['selection']['candidate_id']), None)
    if candidate != expected or candidate['baseline'] != body['baseline_name']:
        raise ValueError('Selected candidate changed after development')
    parents = {candidate['forecaster_id'], candidate['risk_head_id']}
    if set(artifact['parents']) != parents:
        raise ValueError('Incomplete policy predictor/cost lineage')
    for parent in parents:
        if identity['artifacts'].get(parent) != contract.artifacts[parent]:
            raise ValueError('Development producer identity differs from registered parent')
    if file_digest(contract._path(candidate['risk_report_path'])) != candidate['risk_report_sha256']:
        raise ValueError('Cost supervision report changed')
    model = load_verified_forecaster(contract, candidate['forecaster_id'], device=device)
    if model._fitted_baseline_name != candidate['baseline']:
        raise ValueError('Predictor baseline differs from policy baseline')
    return {'id': artifact_id, 'arm': arm, 'baseline': body['baseline_name'], 'model': model,
            'head': _load_cost_head(contract, candidate),
            'settings': contract.protocol['development_evaluation']['policies'][candidate['policy_id']]}


def decide_frozen(scene, policy, *, geometry, solver_seconds, device):
    if policy['arm'] != 'floor':
        result = decide_scene(scene, policy['model'], policy['head'], baseline=policy['baseline'],
                              policy=policy['settings'], geometry=geometry, device=device,
                              solver_seconds=solver_seconds)
        result['arms'] = {policy['id']: result['arms'][policy['arm']]}
        return result
    baseline = np.stack([a['inputs']['baseline_rollouts'][BASELINES.index(policy['baseline'])] for a in scene['agents']])
    return {'agent_ids': [a['agent_id'] for a in scene['agents']], 'baseline': baseline,
            'arms': {policy['id']: {'prediction': baseline.copy(), 'switch': np.zeros(len(baseline), bool),
                     'mean_pair_proxy': 0., 'reason': 'analytical_unchanged_baseline',
                     'predicted_constraints_satisfied': True}}}


def aggregate_bounded_risks(rows_by_policy, policies, *, records, risks, metric, within_scene_aggregation):
    """All decisions stay in the denominator; unknown switched losses get upper=1."""
    policy_ids = [p['id'] for p in policies]
    if len(set(policy_ids)) != len(policy_ids) or set(rows_by_policy) != set(policy_ids) or not rows_by_policy:
        raise ValueError('Complete frozen policy family required')
    scenes = sorted({r['physical_scene'] for r in records.values()})
    values = np.zeros((len(scenes), len(policies), len(risks)))
    coverage, reference = {}, None
    for j, policy in enumerate(policies):
        name, rows = policy['id'], rows_by_policy[policy['id']]
        keys = [(r['recording_id'], r['frame_id'], r['agent_id'], r['horizon_raw'], r['requested_steps']) for r in rows]
        if not rows or len(set(keys)) != len(keys) or {r['recording_id'] for r in rows} != set(records):
            raise ValueError('Each policy needs unique queries and every calibration recording')
        signature = sorted((key, r['scale'], r[f'baseline_{metric}']) for key, r in zip(keys, rows))
        if reference is None:
            reference = signature
        if signature != reference:
            raise ValueError('Calibration policies do not share baseline/query/label eligibility')
        per_row = np.zeros((len(rows), len(risks)))
        unknown_switched = observed = switches = 0
        observed_baseline, observed_policy = [], []
        for i, row in enumerate(rows):
            if records[row['recording_id']]['physical_scene'] != row['physical_scene'] or set(row['arms']) != {name}:
                raise ValueError('Calibration row scene/policy identity mismatch')
            decision = row['arms'][name]
            if not np.isfinite(row['scale']) or row['scale'] <= 0:
                raise ValueError('Finite positive past normalization scale required')
            if type(decision['switch']) is not bool:
                raise ValueError('Explicit Boolean intervention required')
            b, p = row[f'baseline_{metric}'], decision[metric]
            if (b is None) != (p is None) or (b is not None and (not np.isfinite([b, p]).all() or min(b, p) < 0)):
                raise ValueError('Invalid/mismatched bounded-risk labels')
            if policy['arm'] == 'floor' and decision['switch']:
                raise ValueError('Analytical floor may never intervene')
            if not decision['switch'] and b is not None and p != b:
                raise ValueError('Unchanged baseline must have exactly identical loss')
            switches += decision['switch']
            if b is not None:
                observed += 1
                observed_baseline.append(b); observed_policy.append(p)
            if not decision['switch']:
                continue
            if b is None:
                unknown_switched += 1
                per_row[i] = 1.
                continue
            for k, risk in enumerate(risks):
                if risk['statistic'] == 'clipped_positive_excess':
                    per_row[i, k] = min(max(p - b, 0.) / risk['clip_scale'], 1.)
                elif risk['statistic'] == 'harm_event':
                    per_row[i, k] = float(p - b > risk['margin'])
                else:
                    raise ValueError('Unsupported risk functional')
        for s, scene in enumerate(scenes):
            names = sorted(r for r, record in records.items() if record['physical_scene'] == scene)
            parts = [per_row[[i for i, row in enumerate(rows) if row['recording_id'] == r]] for r in names]
            if within_scene_aggregation == 'equal_recording':
                values[s, j] = np.mean([part.mean(0) for part in parts], axis=0)
            elif within_scene_aggregation == 'agent_window':
                values[s, j] = np.concatenate(parts).mean(0)
            else:
                raise ValueError('Unknown within-scene aggregation')
        coverage[name] = {'past_supported_decisions': len(rows), 'observed_primary_labels': observed,
                          'unknown_switched_worst_case': unknown_switched, 'switches': switches,
                          'observed_agent_window_mean_baseline': float(np.mean(observed_baseline)) if observed else None,
                          'observed_agent_window_mean_policy': float(np.mean(observed_policy)) if observed else None,
                          'descriptive_means_not_calibrated_raw_or_easy_ratios': True}
    return scenes, values, coverage


def screen_policies(contract, policies, rows_by_policy):
    rules, recordings = validate_rules(contract)
    if [p['id'] for p in policies] != rules['policy_priority']:
        raise ValueError('Policy order differs from frozen calibration priority')
    risks = contract.protocol['risk']['risks']
    scenes, losses, coverage = aggregate_bounded_risks(rows_by_policy, policies,
        records={r: contract.protocol['records'][r] for r in recordings}, risks=risks,
        metric=contract.protocol['task']['primary_metric'], within_scene_aggregation=rules['within_scene_aggregation'])
    learned = [i for i, p in enumerate(policies) if p['arm'] != 'floor']
    bounds, accepted = np.zeros((len(policies), len(risks))), np.ones(len(policies), bool)
    fitted_scenes = contract.scene_set([r for r, role in contract.protocol['assignments'].items() if role in {'fit', 'development'}])
    if learned:
        result = screen_cluster_risks(losses=losses[:, learned], cluster_ids=scenes,
            fitted_cluster_ids=fitted_scenes, policy_ids=[policies[i]['id'] for i in learned],
            lower=[r['lower'] for r in risks], upper=[r['upper'] for r in risks],
            tolerance=[r['tolerance'] for r in risks], delta=contract.protocol['risk']['delta'])
        bounds[learned], accepted[learned] = result['upper_risk_bound'], result['accepted']
    selected = next((p['id'] for p, ok in zip(policies, accepted) if ok), None)
    return {'protocol_sha256': contract.digest, 'evaluation_role': 'calibration',
            'policy_order': [p['id'] for p in policies], 'risk_specifications': risks,
            'physical_scene_ids': scenes, 'physical_scene_count_declared': len(scenes),
            'statistical_policy_family_size': len(learned), 'risk_count': len(risks),
            'within_scene_aggregation': rules['within_scene_aggregation'], 'between_scene_aggregation': 'equal_physical_scene',
            'scene_losses': losses.tolist(), 'upper_risk_bounds': bounds.tolist(),
            'accepted_under_assumptions': accepted.tolist(), 'coverage': coverage,
            'selected_policy_id': selected, 'use_unchanged_baseline': selected is None or next(p for p in policies if p['id'] == selected)['arm'] == 'floor',
            'selection_rule': rules['selection_rule'],
            'fallback_reason': 'no_policy_passed_bounded_risk_screen' if selected is None else None,
            'risk_scope': 'equal_scene_mean_bounded_loss_under_independent_same_population_scene_assumptions',
            'missing_label_rule': 'worst_case_one_for_switch; exact_zero_excess_for_no_switch',
            'analytical_floor_is_zero_excess_not_zero_error': True,
            'independence_verified': False, 'arbitrary_domain_shift_covered': False,
            'raw_ADE_FDE_or_easy_relative_degradation_certified': False,
            'confirmation_evaluated': False, 'deployment_approved': False,
            'physical_safety_certified': False, 'stage5c_executed': False, 'smc_enabled': False}


def evaluate_calibration(contract, policy_ids, claim_path, *, device='cpu', cached_rows=None,
                         on_recording=None, progress=None):
    rules, recordings = validate_rules(contract)
    if not policy_ids or len(set(policy_ids)) != len(policy_ids):
        raise ValueError('Explicit unique development-prioritized policy IDs required')
    if policy_ids != rules['policy_priority']:
        raise ValueError('Policy order differs from frozen calibration priority')
    _check_claim_path(claim_path, contract, 'calibration')
    _check_claim(json.loads(Path(claim_path).read_text()), contract, policy_ids, 'calibration')
    # Validate every producer before the first calibration reader or label is opened.
    policies = [load_frozen_policy(contract, name, device=device) for name in policy_ids]
    if len({p['baseline'] for p in policies}) != 1:
        raise ValueError('Calibration must compare against one fixed baseline identity')
    all_rows = {}
    for policy in policies:
        rows = []
        for recording in recordings:
            reader, _ = contract.open_recording(recording, purpose='calibration', claim_path=claim_path)
            key = (policy['id'], recording)
            part = (cached_rows or {}).get(key)
            if part is None:
                part = []
                for scene in scene_requests(reader, contract.protocol['task'], stride=rules['query_stride']):
                    decisions = decide_frozen(scene, policy, geometry=rules['geometry_by_recording'][recording],
                                              solver_seconds=rules['solver_seconds'], device=device)
                    labels = reader.get_scene_labels(scene)
                    part.extend(score_scene(scene, decisions, labels, label_policy=rules['label_policy']))
                    if progress:
                        progress(key, scene['frame_id'], len(part))
                if on_recording:
                    on_recording(key, part)
            if any(r['recording_id'] != recording or r['physical_scene'] != reader.metadata['physical_scene'] for r in part):
                raise ValueError('Calibration cache belongs to another recording/scene')
            rows.extend(part)
        all_rows[policy['id']] = rows
    return screen_policies(contract, policies, all_rows), all_rows
