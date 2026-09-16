"""Frozen final-family evaluation; no selection, refitting or deployment promotion."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation.m3w_experiment_contract import file_digest, _claim_identity, _check_claim, _check_claim_path
from src.evaluation.m3w_development_evaluation import (
    ARMS, content_digest, scene_requests, decide_scene, score_scene, summarize_rows,
    _cluster_contributions, _load_cost_head, _validate_policy,
)
from src.evaluation.m3w_risk_calibration import load_frozen_policy
from src.world_model.m3w_supervised_intervention import load_verified_forecaster

ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    'src/evaluation/m3w_confirmation_evaluation.py', 'scripts/evaluate_m3w_confirmation.py',
    'src/evaluation/m3w_risk_calibration.py', 'scripts/calibrate_m3w_intervention.py',
    'src/evaluation/m3w_development_evaluation.py', 'src/evaluation/m3w_experiment_contract.py',
    'src/data_unification/m3w_causal_recordings.py', 'src/world_model/m3w_supervised_intervention.py',
    'src/world_model/m3w_joint_intervention.py',
)


def implementation_identity():
    return {name: file_digest(ROOT / name) for name in CODE_PATHS}


def query_key(row):
    return tuple(row[k] for k in ('recording_id', 'frame_id', 'agent_id', 'horizon_raw', 'requested_steps'))


def validate_rules(contract):
    contract._assert_frozen()
    rules = contract.protocol.get('confirmation_evaluation', {})
    required = {'error_unit', 'label_policy', 'query_stride', 'geometry_by_recording', 'easy_threshold',
                'hard_threshold', 'solver_seconds', 'bootstrap_seed', 'baseline', 'comparisons', 'calibrator_id'}
    if set(rules) != required or rules['error_unit'] != 'past_normalized' or rules['label_policy'] != 'complete_requested_path':
        raise ValueError('Explicit approved complete-path normalized confirmation rules required')
    if (type(rules['query_stride']) is not int or rules['query_stride'] < 1
            or type(rules['bootstrap_seed']) is not int
            or not np.isfinite([rules['easy_threshold'], rules['hard_threshold'], rules['solver_seconds']]).all()
            or not 0 <= rules['easy_threshold'] < rules['hard_threshold'] or rules['solver_seconds'] <= 0):
        raise ValueError('Explicit confirmation query, slice and bootstrap settings required')
    if rules['baseline'] not in BASELINES:
        raise ValueError('Explicit supported frozen baseline required')
    easy = {'kind': 'baseline_error_at_most', 'metric': contract.protocol['task']['primary_metric'],
            'error_unit': rules['error_unit'], 'threshold': rules['easy_threshold']}
    if easy != contract.protocol['risk']['easy_definition']:
        raise ValueError('Confirmation easy slice differs from approved risk definition')
    recordings = sorted(r for r, role in contract.protocol['assignments'].items() if role == 'confirmation')
    if set(rules['geometry_by_recording']) != set(recordings):
        raise ValueError('Explicit geometry for every confirmation recording required')
    for geometry in rules['geometry_by_recording'].values():
        if (set(geometry) != {'graph_radius', 'proximity_threshold'}
                or not np.isfinite(list(geometry.values())).all() or min(geometry.values()) <= 0):
            raise ValueError('Invalid recording-local proximity geometry')
    candidates = rules['comparisons']
    fields = {'id', 'family', 'seed', 'forecaster_id', 'risk_head_id', 'risk_report_path', 'policy_id'}
    if not candidates or any(set(c) != fields for c in candidates):
        raise ValueError('Explicit complete confirmation comparison family required')
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate comparison identity')
    groups = {}
    for c in candidates:
        if any(not isinstance(c[k], str) or not c[k] for k in fields - {'seed'}) or type(c['seed']) is not int:
            raise ValueError('Invalid comparison identity/seed')
        groups.setdefault(c['family'], []).append(c['seed'])
        policy = contract.protocol['development_evaluation']['policies'].get(c['policy_id'])
        if policy is None:
            raise ValueError('Confirmation policy was not in the approved development grid')
        _validate_policy(policy)
    if any(sorted(seeds) != sorted(contract.protocol['seeds']) for seeds in groups.values()):
        raise ValueError('Every comparison family needs exactly all protocol training seeds')
    ids = sorted({rules['calibrator_id'], *(c[k] for c in candidates for k in ('forecaster_id', 'risk_head_id'))})
    for name in ids:
        contract.assert_prediction_use(name, recordings, purpose='confirmation')
    return rules, recordings, ids


def load_calibration_selection(contract, calibrator_id, *, device):
    """Check the completed calibration and its frozen policy exports without labels."""
    record = contract.artifacts[calibrator_id]
    calibration = sorted(r for r, role in contract.protocol['assignments'].items() if role == 'calibration')
    priority = contract.protocol['calibration_evaluation']['policy_priority']
    if (record['kind'] != 'calibrator' or record['fit_recordings'] or record['selection_recordings']
            or sorted(record['calibration_recordings']) != calibration or record['parents'] != priority):
        raise ValueError('Incomplete calibration lineage')
    path = contract._path(record['path'])
    report = json.loads(path.read_text())
    directory = path.parent
    identity = json.loads((directory / 'run_identity.json').read_text())
    completion = json.loads((directory / 'completion.json').read_text())
    claim = json.loads(contract._path(contract.protocol['calibration_receipt']).read_text())
    expected = _claim_identity(contract, priority, 'calibration')
    if (any(claim.get(k) != v for k, v in expected.items()) or claim['state'] != 'completed'
            or Path(claim['result_path']).resolve() != path or claim['result_sha256'] != record['sha256']
            or identity['protocol_sha256'] != contract.digest or identity['policy_order'] != priority
            or completion['run_sha256'] != content_digest(identity)):
        raise ValueError('Calibration completion/claim identity mismatch')
    for name, digest in identity['code_sha256'].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or file_digest(source) != digest:
            raise ValueError('Calibration implementation changed')
    if set(completion['artifacts']) != {'calibration_report.json', 'artifact.json'}:
        raise ValueError('Incomplete calibration result receipt')
    for name, digest in completion['artifacts'].items():
        if file_digest(directory / name) != digest:
            raise ValueError('Calibration result changed')
    if json.loads((directory / 'artifact.json').read_text()) != record or path.name != 'calibration_report.json':
        raise ValueError('Calibration artifact differs from completion')
    if (report['protocol_sha256'] != contract.digest or report['evaluation_role'] != 'calibration'
            or report['policy_order'] != priority or report['risk_specifications'] != contract.protocol['risk']['risks']):
        raise ValueError('Calibration family/risk mismatch')
    policies = {name: load_frozen_policy(contract, name, device=device) for name in priority}
    accepted = report['accepted_under_assumptions']
    if len(accepted) != len(priority) or any(type(a) is not bool for a in accepted):
        raise ValueError('Invalid calibrated acceptance flags')
    selected = next((name for name, ok in zip(priority, accepted) if ok), None)
    floor = selected is None or policies[selected]['arm'] == 'floor'
    if report['selected_policy_id'] != selected or report['use_unchanged_baseline'] != floor:
        raise ValueError('Calibration selection does not follow frozen priority')
    return report, policies.get(selected), policies


def load_family(contract, *, device):
    rules, recordings, ids = validate_rules(contract)
    calibrated, selected, policies = load_calibration_selection(contract, rules['calibrator_id'], device=device)
    if {p['baseline'] for p in policies.values()} != {rules['baseline']}:
        raise ValueError('Calibrated floor differs from comparison floor')
    loaded = []
    for c in rules['comparisons']:
        # Check actual producer seeds, not just names supplied in a result table.
        ancestors = contract._closure(c['risk_head_id']) | {c['forecaster_id']}
        for name in ancestors:
            artifact = contract.artifacts[name]
            if artifact['kind'] == 'forecaster':
                state = torch.load(contract._path(artifact['path']), map_location='cpu', weights_only=True)
                if state['identity']['settings']['seed'] != c['seed']:
                    raise ValueError('Claimed seed differs from forecaster/OOF producer training seed')
        model = load_verified_forecaster(contract, c['forecaster_id'], device=device)
        if model._fitted_baseline_name != rules['baseline']:
            raise ValueError('Predictor and evaluation baseline disagree')
        candidate = {**c, 'baseline': rules['baseline']}
        head = _load_cost_head(contract, candidate)
        loaded.append({**candidate, 'model': model, 'head': head,
                       'settings': contract.protocol['development_evaluation']['policies'][c['policy_id']]})
    reference = None
    if not calibrated['use_unchanged_baseline']:
        body = json.loads(contract._path(contract.artifacts[selected['id']]['path']).read_text())['candidate']
        reference = next((c['id'] for c in loaded if all(c[k] == body[k] for k in
                         ('forecaster_id', 'risk_head_id', 'policy_id', 'risk_report_path', 'baseline'))), None)
        if reference is None:
            raise ValueError('Calibrated selected producer absent from frozen comparison family')
    return rules, recordings, ids, loaded, {'calibrator_id': rules['calibrator_id'],
        'policy_id': calibrated['selected_policy_id'], 'comparison_id': reference,
        'arm': 'floor' if reference is None else selected['arm'],
        'calibration_independence_verified': calibrated['independence_verified'], 'deployment_approved': False}


def _support_signature(rows):
    return sorted((query_key(r), r['physical_scene'], r['scale'], r['baseline_ade'], r['baseline_fde'],
                   r['available_label_steps']) for r in rows)


def _final_summary(rows, contract, rules):
    result = summarize_rows(rows, metric=contract.protocol['task']['primary_metric'],
        aggregation=contract.protocol['task']['aggregation'], error_unit=rules['error_unit'],
        easy_threshold=rules['easy_threshold'], hard_threshold=rules['hard_threshold'],
        bootstrap_resamples=contract.protocol['bootstrap_resamples'], bootstrap_seed=rules['bootstrap_seed'])
    # Keep these descriptive intervals separate from calibrated bounds or adjusted tests.
    def annotate(value):
        if isinstance(value, dict):
            if value.get('status') == 'computed_development_descriptive_not_confirmation':
                value['status'] = 'computed_fixed_family_scene_bootstrap_descriptive'
                value['multiple_comparisons_adjusted'] = False
            for item in value.values():
                annotate(item)
    annotate(result)
    metric, aggregation = (contract.protocol['task'][k] for k in ('primary_metric', 'aggregation'))
    def update_worst(destination, part, arm, m):
        valid = [r for r in part if r[f'baseline_{m}'] is not None]
        if valid:
            _, sums, weights = _cluster_contributions(valid, np.asarray([[r['arms'][arm][m]] for r in valid]), aggregation)
            destination['worst_physical_scene_mean_error'] = float(np.max(sums[:, 0] / weights))

    for arm in result['arms']:
        eligible = [r for r in rows if r[f'baseline_{metric}'] is not None]
        slices = {'all': eligible,
                  'easy': [r for r in eligible if r[f'baseline_{metric}'] <= rules['easy_threshold']],
                  'hard': [r for r in eligible if r[f'baseline_{metric}'] >= rules['hard_threshold']]}
        for name, part in slices.items():
            update_worst(result['arms'][arm][name], part, arm, metric)
        for section, key in (('per_physical_scene', 'physical_scene'), ('per_raw_horizon', 'horizon_raw')):
            for name, destination in result['arms'][arm][section].items():
                update_worst(destination, [r for r in rows if str(r[key]) == name], arm, metric)
        secondary = 'fde' if metric == 'ade' else 'ade'
        update_worst(result['arms'][arm]['secondary_' + secondary], rows, arm, secondary)
    result['confirmation_result'] = True
    result['independent_confirmation_verified'] = False
    result['eligibility_scope'] = contract.protocol['scope']
    return result


def paired_arm_summary(rows, left, right, *, contract, rules):
    """Paired scene bootstrap; negative difference favors left. Not a safety bound."""
    metric, aggregation = (contract.protocol['task'][k] for k in ('primary_metric', 'aggregation'))
    rows = [r for r in rows if r[f'baseline_{metric}'] is not None]
    if not rows:
        return {'status': 'not_run_no_eligible_labels'}
    values = np.asarray([[r['arms'][left][metric], r['arms'][right][metric]] for r in rows])
    scenes, sums, weights = _cluster_contributions(rows, values, aggregation)
    mean = sums.sum(0) / weights.sum()
    result = {'left': left, 'right': right, 'mean_error_difference': float(mean[0]-mean[1]),
              'physical_scene_count': len(scenes), 'resampling_unit': 'physical_scene',
              'ci95': None, 'multiple_comparisons_adjusted': False, 'independence_verified': False,
              'status': 'not_run_insufficient_physical_scenes_for_interval'}
    if len(scenes) >= 2:
        draws = np.random.default_rng(rules['bootstrap_seed']).integers(len(scenes), size=(contract.protocol['bootstrap_resamples'], len(scenes)))
        means = sums[draws].sum(1) / weights[draws].sum(1)[:, None]
        result.update(status='computed_paired_scene_descriptive',
                      ci95=np.quantile(means[:, 0]-means[:, 1], [.025, .975]).tolist())
    return result


def summarize_family(contract, rules, all_rows, calibrated_reference):
    candidates = rules['comparisons']
    for family in {c['family'] for c in candidates}:
        if sorted(c['seed'] for c in candidates if c['family'] == family) != sorted(contract.protocol['seeds']):
            raise ValueError('Every comparison family needs exactly all protocol training seeds')
    if set(all_rows) != {c['id'] for c in candidates}:
        raise ValueError('Incomplete final comparison family')
    reference = None
    for c in candidates:
        rows = all_rows[c['id']]
        if not rows or len({query_key(r) for r in rows}) != len(rows):
            raise ValueError('Empty or duplicate final queries')
        if any(set(r['arms']) != set(ARMS) for r in rows):
            raise ValueError('Final comparisons require all five matched controls')
        signature = _support_signature(rows)
        if reference is not None and signature != reference:
            raise ValueError('Seeds/families differ in confirmation support or baseline')
        reference = signature
        for r in rows:
            if contract.protocol['assignments'].get(r['recording_id']) != 'confirmation':
                raise ValueError('Final summary contains a non-confirmation recording')
            if r['physical_scene'] != contract.protocol['records'][r['recording_id']]['physical_scene']:
                raise ValueError('Confirmation physical-scene identity mismatch')
            if not np.isfinite(r['scale']) or r['scale'] <= 0:
                raise ValueError('Invalid past normalization scale')
            for m in ('ade', 'fde'):
                if r['arms']['floor'][m] != r[f'baseline_{m}']:
                    raise ValueError('Floor error differs from baseline')
            for arm in ARMS:
                choice = r['arms'][arm]
                if type(choice['switch']) is not bool or not np.isfinite(choice['pair_proxy']):
                    raise ValueError('Invalid decision/proximity result')
                if arm == 'floor' and choice['switch']:
                    raise ValueError('Floor cannot intervene')
                if not choice['switch'] and any(choice[m] != r[f'baseline_{m}'] for m in ('ade', 'fde')):
                    raise ValueError('Unchanged decision changed prediction loss')
    per_seed = {c['id']: _final_summary(all_rows[c['id']], contract, rules) for c in candidates}
    families = {}
    for family in sorted({c['family'] for c in candidates}):
        parts = [c for c in candidates if c['family'] == family]
        aligned = [sorted(all_rows[c['id']], key=query_key) for c in parts]
        mean_rows = deepcopy(aligned[0])
        for i, row in enumerate(mean_rows):
            for arm in ARMS:
                source = [part[i]['arms'][arm] for part in aligned]
                for metric in ('ade', 'fde', 'pair_proxy', 'switch'):
                    row['arms'][arm][metric] = (None if source[0][metric] is None else
                        float(np.mean([r[metric] for r in source])))
                row['arms'][arm]['reason'] = 'summary_of_fixed_seeds_not_an_ensemble_prediction'
                row['arms'][arm]['constraints_satisfied'] = all(r['constraints_satisfied'] for r in source)
        summary = _final_summary(mean_rows, contract, rules)
        summary.update(training_seeds=[c['seed'] for c in parts],
            estimand='mean_error_of_fixed_training_seeds_not_error_of_ensemble_mean_trajectory',
            bootstrap_conditions_on_fixed_seeds=True, seeds_do_not_increase_independent_scene_n=True,
            tail_scope='agent_window_p95_of_seed_mean_error; per_seed_tails_reported_separately')
        summary['seed_variability'] = {}
        summary['empirical_easy_guard_by_seed'] = {}
        for arm in ARMS:
            # Positive harm is nonlinear: average each seed's positive part,
            # never the positive part of its mean, which would hide seed harm.
            def average_harm(destination, sources):
                if destination.get('status') == 'evaluated':
                    destination['mean_positive_harm'] = float(np.mean([s['mean_positive_harm'] for s in sources]))

            for slice_name in ('all', 'easy', 'hard', 'secondary_' + ('fde' if contract.protocol['task']['primary_metric'] == 'ade' else 'ade')):
                average_harm(summary['arms'][arm][slice_name], [per_seed[c['id']]['arms'][arm][slice_name] for c in parts])
            for section in ('per_physical_scene', 'per_raw_horizon'):
                for name, destination in summary['arms'][arm][section].items():
                    average_harm(destination, [per_seed[c['id']]['arms'][arm][section][name] for c in parts])
            summary['seed_variability'][arm] = {}
            summary['empirical_easy_guard_by_seed'][arm] = {}
            for c in parts:
                easy = per_seed[c['id']]['arms'][arm]['easy']
                check = None
                if easy['status'] == 'evaluated':
                    check = (easy['selected_error'] <= easy['baseline_error'] if easy['degradation_fraction'] is None
                             else easy['degradation_fraction'] <= contract.protocol['risk']['easy_degradation_max'])
                summary['empirical_easy_guard_by_seed'][arm][str(c['seed'])] = bool(check) if check is not None else None
            for slice_name in ('all', 'easy', 'hard'):
                metrics = [per_seed[c['id']]['arms'][arm][slice_name] for c in parts]
                errors = [m.get('selected_error') for m in metrics]
                gains = [m.get('improvement_pct') for m in metrics]
                summary['seed_variability'][arm][slice_name] = {
                    'selected_error_by_seed': errors, 'improvement_pct_by_seed': gains,
                    'selected_error_sample_std': float(np.std(errors, ddof=1)) if all(e is not None for e in errors) else None,
                    'improvement_pct_sample_std': float(np.std(gains, ddof=1)) if all(g is not None for g in gains) else None}
        summary['paired_joint_controls'] = {other: paired_arm_summary(mean_rows, 'joint', other, contract=contract, rules=rules)
                                            for other in ('floor', 'uncontrolled', 'independent', 'scene_uniform')}
        families[family] = summary
    chosen = calibrated_reference['comparison_id'] or candidates[0]['id']
    selected_arm = calibrated_reference['arm']
    floor_summary = per_seed[chosen]
    calibrated = {**calibrated_reference, 'metrics': floor_summary['arms'][selected_arm],
                  'interpretation': 'one_calibration_selected_policy_not_a_three_seed_ensemble',
                  'dataset_local_metrics': {name: {'baseline': value['baseline'], 'selected': value['arms'][selected_arm]}
                        for name, value in floor_summary['dataset_local_metrics']['recordings'].items()}}
    return {'evaluation_role': 'confirmation', 'protocol_sha256': contract.digest,
            'result_source': 'fresh_run', 'scope': contract.protocol['scope'], 'per_seed': per_seed,
            'families': families, 'frozen_calibrated_policy': calibrated,
            'nonlinear_harm_aggregation': 'mean_of_per_seed_positive_harm_not_positive_of_seed_mean_excess',
            'model_selection_performed': False, 'deployment_approved': False,
            'independent_confirmation_eligibility_by_declaration': contract.confirmation_eligible_by_declaration,
            'independent_confirmation_verified': False, 'physical_safety_certified': False,
            'strongest_baseline_claim': False, 'stage5c_executed': False, 'smc_enabled': False}


def evaluate_confirmation(contract, claim_path, *, device='cpu', cached_rows=None, on_recording=None, progress=None):
    rules, recordings, ids, family, reference = load_family(contract, device=device)
    _check_claim_path(claim_path, contract, 'confirmation')
    claim = json.loads(Path(claim_path).read_text())
    _check_claim(claim, contract, ids, 'confirmation')
    if claim.get('implementation_sha256') != implementation_identity():
        raise ValueError('Confirmation claim must freeze the evaluated implementation')
    all_rows = {}
    for candidate in family:
        rows = []
        for recording in recordings:
            reader, _ = contract.open_recording(recording, purpose='confirmation', claim_path=claim_path)
            key = (candidate['id'], recording)
            part = (cached_rows or {}).get(key)
            if part is None:
                part = []
                for scene in scene_requests(reader, contract.protocol['task'], stride=rules['query_stride']):
                    decisions = decide_scene(scene, candidate['model'], candidate['head'], baseline=rules['baseline'],
                        policy=candidate['settings'], geometry=rules['geometry_by_recording'][recording],
                        device=device, solver_seconds=rules['solver_seconds'])
                    labels = reader.get_scene_labels(scene)
                    part.extend(score_scene(scene, decisions, labels, label_policy=rules['label_policy']))
                    if progress:
                        progress(key, scene['frame_id'], len(part))
                if on_recording:
                    on_recording(key, part)
            if any(r['recording_id'] != recording or r['physical_scene'] != reader.metadata['physical_scene'] for r in part):
                raise ValueError('Final cache recording/scene identity mismatch')
            rows.extend(part)
        all_rows[candidate['id']] = rows
    return summarize_family(contract, rules, all_rows, reference), all_rows
