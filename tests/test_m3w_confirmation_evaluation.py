from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_m3w_development_evaluation import GEOMETRY, POLICY
from test_m3w_experiment_contract import approve
from test_m3w_supervised_intervention import datasets, training_config
from test_m3w_risk_calibration import RISKS
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, claim_confirmation
from src.evaluation.m3w_development_evaluation import ARMS
from src.evaluation.m3w_confirmation_evaluation import (
    load_family, validate_rules, implementation_identity, summarize_family, evaluate_confirmation,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def comparisons():
    return [{'id': f's{s}', 'family': 'transformer', 'seed': s, 'forecaster_id': f's{s}_fit_ab',
             'risk_head_id': f'cost_{s}', 'risk_report_path': f'cost_{s}.json', 'policy_id': 'p1'} for s in (1, 2, 3)]


def summary_fixture():
    p = {'task': {'primary_metric': 'ade', 'aggregation': 'equal_physical_scene'}, 'bootstrap_resamples': 2000,
         'risk': {'easy_degradation_max': .02},
         'seeds': [1, 2, 3], 'scope': 'confirmatory', 'records': {'t1': {'physical_scene': 'one'}, 't2': {'physical_scene': 'two'}},
         'assignments': {'t1': 'confirmation', 't2': 'confirmation'}}
    contract = SimpleNamespace(protocol=p, digest='synthetic', confirmation_eligible_by_declaration=True)
    rules = {'comparisons': comparisons(), 'error_unit': 'past_normalized', 'easy_threshold': .5,
             'hard_threshold': 2., 'bootstrap_seed': 11}
    all_rows = {}
    for s in (1, 2, 3):
        rows = []
        for recording, scene, baseline in (('t1', 'one', .4), ('t2', 'two', 4.)):
            for agent in (1, 2):
                r = {'recording_id': recording, 'physical_scene': scene, 'frame_id': 80, 'agent_id': agent,
                     'horizon_raw': 120, 'requested_steps': 12, 'available_label_steps': 12,
                     'scale': 1., 'baseline_ade': baseline, 'baseline_fde': baseline, 'arms': {}}
                for arm in ARMS:
                    factor = 1. if arm == 'floor' else (.4 + s*.1 if arm == 'joint' else .9)
                    r['arms'][arm] = {'ade': baseline*factor, 'fde': baseline*factor, 'switch': arm != 'floor',
                        'pair_proxy': 0., 'reason': 'synthetic', 'constraints_satisfied': True}
                rows.append(r)
        all_rows[f's{s}'] = rows
    reference = {'calibrator_id': 'cal', 'policy_id': 'dev', 'comparison_id': None, 'arm': 'floor',
                 'calibration_independence_verified': False, 'deployment_approved': False}
    return contract, rules, all_rows, reference


def test_fixed_seed_mean_and_paired_scene_ci_not_pseudoreplication():
    contract, rules, rows, reference = summary_fixture()
    result = summarize_family(contract, rules, rows, reference)
    json.dumps(result, allow_nan=False)
    family = result['families']['transformer']
    assert family['agent_query_count'] == 4 and family['physical_scene_count'] == 2
    assert family['training_seeds'] == [1, 2, 3]
    assert family['arms']['joint']['all']['improvement_pct'] == pytest.approx(40.)
    assert family['seed_variability']['joint']['all']['improvement_pct_sample_std'] == pytest.approx(10.)
    assert family['paired_joint_controls']['independent']['ci95'][1] < 0
    assert family['arms']['joint']['all']['bootstrap']['status'] == 'computed_fixed_family_scene_bootstrap_descriptive'
    assert not result['model_selection_performed'] and not result['deployment_approved']
    assert result['frozen_calibrated_policy']['metrics']['all']['improvement_pct'] == 0


@pytest.mark.parametrize('violation', ['missing_seed', 'duplicate_seed', 'different_support', 'different_baseline',
                                     'wrong_role', 'floor_switch', 'unmatched_loss', 'duplicate_query'])
def test_final_family_rejects_invalid_matching(violation):
    contract, rules, rows, reference = summary_fixture()
    if violation == 'missing_seed':
        rules['comparisons'].pop()
    elif violation == 'duplicate_seed':
        rules['comparisons'][-1]['seed'] = 1
    elif violation == 'different_support':
        rows['s2'].pop()
    elif violation == 'different_baseline':
        rows['s2'][0]['baseline_ade'] = .9
    elif violation == 'wrong_role':
        contract.protocol['assignments']['t2'] = 'development'
    elif violation == 'floor_switch':
        rows['s1'][0]['arms']['floor']['switch'] = True
    elif violation == 'unmatched_loss':
        rows['s1'][0]['arms']['joint']['ade'] = None
    else:
        rows['s1'].append(deepcopy(rows['s1'][0]))
    with pytest.raises(ValueError):
        summarize_family(contract, rules, rows, reference)


def test_one_scene_does_not_gain_ci_by_adding_training_seeds():
    contract, rules, rows, reference = summary_fixture()
    rows = {key: [r for r in values if r['recording_id'] == 't1'] for key, values in rows.items()}
    result = summarize_family(contract, rules, rows, reference)['families']['transformer']
    assert result['arms']['joint']['all']['bootstrap']['status'] == 'not_run_insufficient_physical_scenes'
    assert result['paired_joint_controls']['floor']['ci95'] is None
    assert result['seed_variability']['joint']['all']['improvement_pct_sample_std'] > 0


def test_harm_is_not_cancelled_between_good_and_bad_seeds():
    contract, rules, rows, reference = summary_fixture()
    for s, factor in ((1, 1.3), (2, .7), (3, 1.)):
        for row in rows[f's{s}']:
            row['arms']['joint']['ade'] = factor * row['baseline_ade']
    result = summarize_family(contract, rules, rows, reference)['families']['transformer']
    assert result['arms']['joint']['all']['mean_excess_over_floor'] == pytest.approx(0.)
    assert result['arms']['joint']['all']['mean_positive_harm'] == pytest.approx(.22)
    assert result['empirical_easy_guard_by_seed']['joint'] == {'1': False, '2': True, '3': True}


def test_worst_scene_respects_equal_recording_not_window_weighting():
    contract, rules, rows, reference = summary_fixture()
    contract.protocol['task']['aggregation'] = 'equal_recording'
    contract.protocol['records'] = {r: {'physical_scene': s} for r, s in [('t1', 'one'), ('t2', 'one'), ('t3', 'two')]}
    contract.protocol['assignments'] = {r: 'confirmation' for r in contract.protocol['records']}
    template = rows['s1'][0]
    parts = []
    for rec, scene, count, error in [('t1', 'one', 1, 1.), ('t2', 'one', 9, 9.), ('t3', 'two', 2, 5.)]:
        for agent in range(count):
            row = deepcopy(template)
            row.update(recording_id=rec, physical_scene=scene, agent_id=agent, baseline_ade=10., baseline_fde=10.)
            for arm in ARMS:
                row['arms'][arm].update(ade=10. if arm == 'floor' else error, fde=10. if arm == 'floor' else error)
            parts.append(row)
    result = summarize_family(contract, rules, {f's{s}': deepcopy(parts) for s in (1, 2, 3)}, reference)
    for summary in [*result['per_seed'].values(), result['families']['transformer']]:
        assert summary['arms']['joint']['all']['worst_physical_scene_mean_error'] == pytest.approx(5.)
        assert summary['arms']['joint']['secondary_fde']['worst_physical_scene_mean_error'] == pytest.approx(5.)


def test_zero_denominator_ratio_is_not_reported_as_safe_improvement():
    contract, rules, rows, reference = summary_fixture()
    for part in rows.values():
        for row in part:
            for metric in ('ade', 'fde'):
                row[f'baseline_{metric}'] = 0.
                row['arms']['floor'][metric] = 0.
    result = summarize_family(contract, rules, rows, reference)['families']['transformer']['arms']['joint']['easy']
    assert result['improvement_pct'] is None and result['mean_excess_over_floor'] > 0
    assert result['bootstrap']['improvement_pct_ci95'] is None


@pytest.fixture
def trained_final_fixture(tmp_path):
    from src.world_model.m3w_supervised_intervention import (
        ContractForecastDataset, train_forecaster, load_verified_forecaster, make_oof_cost_rows, fit_linear_gain_harm,
    )
    initial, _, _ = datasets(tmp_path)
    p = deepcopy(initial.protocol)
    p['risk']['risks'] = deepcopy(RISKS)
    p['risk']['easy_definition'] = {'kind': 'baseline_error_at_most', 'metric': 'ade', 'error_unit': 'past_normalized', 'threshold': .2}
    p['development_evaluation'] = {'error_unit': 'past_normalized', 'label_policy': 'complete_requested_path',
        'query_stride': 2, 'geometry_by_recording': {'d': GEOMETRY}, 'easy_threshold': .2, 'hard_threshold': .5,
        'eligible_arms': ['independent', 'scene_uniform', 'joint'], 'solver_seconds': 2., 'bootstrap_seed': 11, 'policies': {'p1': POLICY}}
    p['calibration_evaluation'] = {'error_unit': 'past_normalized', 'label_policy': 'complete_requested_path',
        'query_stride': 2, 'geometry_by_recording': {'c': GEOMETRY}, 'solver_seconds': 2.,
        'within_scene_aggregation': 'agent_window', 'selection_rule': 'first_accepted_in_frozen_development_order', 'policy_priority': ['dev']}
    p['confirmation_evaluation'] = {'error_unit': 'past_normalized', 'label_policy': 'complete_requested_path',
        'query_stride': 2, 'geometry_by_recording': {'t': GEOMETRY}, 'solver_seconds': 2., 'bootstrap_seed': 11,
        'easy_threshold': .2, 'hard_threshold': .5, 'baseline': 'constant_velocity_causal_fd',
        'comparisons': comparisons(), 'calibrator_id': 'cal'}
    approve(p)
    base = 'constant_velocity_causal_fd'
    artifacts, plan = [], {'schema_version': 1, 'candidates': []}
    contract = ExperimentContract(p, tmp_path)
    architecture, settings = training_config()
    for seed in (1, 2, 3):
        for suffix, names in (('fit_a', ['a']), ('fit_b', ['b']), ('fit_ab', ['a', 'b'])):
            name = f's{seed}_{suffix}'
            data = ContractForecastDataset(contract, names, purpose='fit', baseline_name=base)
            output = train_forecaster(data, architecture=architecture, settings={**settings, 'seed': seed}, output_dir=tmp_path / name)
            path = Path(output['checkpoint'])
            artifacts.append({'id': name, 'kind': 'forecaster', 'path': str(path.relative_to(tmp_path)), 'sha256': file_digest(path),
                'protocol_sha256': contract.digest, 'lineage_complete': True, 'fit_recordings': names,
                'selection_recordings': [], 'calibration_recordings': [], 'parents': []})
        contract = ExperimentContract(p, tmp_path, artifacts)
        groups = []
        for suffix, held in (('fit_a', 'b'), ('fit_b', 'a')):
            name = f's{seed}_{suffix}'
            data = ContractForecastDataset(contract, [held], purpose='fit', baseline_name=base)
            model = load_verified_forecaster(contract, name, device='cpu')
            groups.append(make_oof_cost_rows(contract, name, data, model, batch_size=8, device='cpu'))
        head = fit_linear_gain_harm(contract, groups, alpha=1.)
        path = tmp_path / f'cost_{seed}.npz'
        np.savez(path, **{k: head[k] for k in ('mean', 'scale', 'coef', 'intercept')})
        report = {k: v for k, v in head.items() if k not in {'mean', 'scale', 'coef', 'intercept'}}
        report.update(checkpoint_sha256=file_digest(path), code_sha256=file_digest(ROOT / 'src/world_model/m3w_supervised_intervention.py'))
        report_path = tmp_path / f'cost_{seed}.json'
        report_path.write_text(json.dumps(report))
        artifacts.append({'id': f'cost_{seed}', 'kind': 'risk_head', 'path': path.name, 'sha256': file_digest(path),
            'protocol_sha256': contract.digest, 'lineage_complete': True, 'fit_recordings': ['a', 'b'], 'selection_recordings': [],
            'calibration_recordings': [], 'parents': [f's{seed}_fit_a', f's{seed}_fit_b']})
        c = next(c for c in comparisons() if c['seed'] == seed)
        plan['candidates'].append({k: v for k, v in {**c, 'baseline': base, 'risk_report_sha256': file_digest(report_path)}.items()
                                   if k not in {'family', 'seed'}})
        contract = ExperimentContract(p, tmp_path, artifacts)
    paths = []
    for a in artifacts:
        path = tmp_path / f'{a["id"]}.artifact.json'
        path.write_text(json.dumps(a)); paths.append(path)
    ppath, planpath = tmp_path / 'protocol.json', tmp_path / 'plan.json'
    ppath.write_text(json.dumps(p)); planpath.write_text(json.dumps(plan))
    common = ['--protocol', str(ppath), '--workspace-root', str(tmp_path), '--threads', '2']
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_m3w_development.py'), *common,
        '--plan', str(planpath), '--artifacts', *map(str, paths), '--output-dir', str(tmp_path / 'dev')],
        check=True, capture_output=True, timeout=90)
    paths.append(tmp_path / 'dev/artifact.json')
    subprocess.run([sys.executable, str(ROOT / 'scripts/calibrate_m3w_intervention.py'), *common,
        '--artifacts', *map(str, paths), '--policy-ids', 'dev', '--output-dir', str(tmp_path / 'cal')],
        check=True, capture_output=True, timeout=90)
    paths.append(tmp_path / 'cal/artifact.json')
    return ExperimentContract(p, tmp_path, [json.loads(path.read_text()) for path in paths]), paths, common


def reserve(contract):
    path = contract._path(contract.protocol['confirmation_receipt'])
    _, _, ids = validate_rules(contract)
    claim = claim_confirmation(path, contract, ids)
    claim['implementation_sha256'] = implementation_identity()
    path.write_text(json.dumps(claim))
    return path


def test_real_three_seed_pipeline_final_role_and_no_selection(trained_final_fixture, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    import src.evaluation.m3w_confirmation_evaluation as module
    contract, _, _ = trained_final_fixture
    events = []
    original_open, original_decide, original_labels = contract.open_recording, module.decide_scene, RecordingWindows.get_scene_labels

    def opened(name, **kwargs):
        assert name == 't' and kwargs['purpose'] == 'confirmation'
        return original_open(name, **kwargs)

    def decided(scene, *args, **kwargs):
        value = original_decide(scene, *args, **kwargs)
        events.append(('decision', scene['frame_id']))
        return value

    def labeled(reader, scene):
        assert reader.metadata['id'] == 't' and events[-1] == ('decision', scene['frame_id'])
        events.append(('label', scene['frame_id']))
        return original_labels(reader, scene)

    monkeypatch.setattr(contract, 'open_recording', opened)
    monkeypatch.setattr(module, 'decide_scene', decided)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', labeled)
    before = {name: file_digest(contract._path(a['path'])) for name, a in contract.artifacts.items()}
    result, rows = evaluate_confirmation(contract, reserve(contract))
    assert events and set(rows) == {'s1', 's2', 's3'}
    assert result['families']['transformer']['training_seeds'] == [1, 2, 3]
    assert not result['model_selection_performed'] and not result['deployment_approved']
    assert result['families']['transformer']['physical_scene_count'] == 1
    assert before == {name: file_digest(contract._path(a['path'])) for name, a in contract.artifacts.items()}


def test_seed_claims_checked_against_forecaster_and_oof_producers(trained_final_fixture):
    contract, _, _ = trained_final_fixture
    # A renamed checkpoint with the wrong seed cannot provide seed replication.
    p = deepcopy(contract.protocol)
    a, b = p['confirmation_evaluation']['comparisons'][:2]
    a['seed'], b['seed'] = b['seed'], a['seed']
    contract.protocol = p
    # Exercise the producer check after validating the original approved metadata.
    contract._assert_frozen = lambda: None
    import src.evaluation.m3w_confirmation_evaluation as module
    original = module.load_calibration_selection
    try:
        module.load_calibration_selection = lambda *_a, **_k: ({'use_unchanged_baseline': True, 'selected_policy_id': None, 'independence_verified': False}, None,
            {'dev': {'baseline': p['confirmation_evaluation']['baseline']}})
        with pytest.raises(ValueError, match='producer training seed'):
            load_family(contract, device='cpu')
    finally:
        module.load_calibration_selection = original


def test_confirmation_requires_exact_frozen_claim_before_labels(trained_final_fixture, monkeypatch):
    contract, _, _ = trained_final_fixture
    monkeypatch.setattr(contract, 'open_recording', lambda *_a, **_k: pytest.fail('opened before valid claim'))
    with pytest.raises(FileNotFoundError):
        evaluate_confirmation(contract, contract.root / 'confirmation.json')
    path = reserve(contract)
    claim = json.loads(path.read_text()); claim['implementation_sha256'] = {}
    path.write_text(json.dumps(claim))
    with pytest.raises(ValueError, match='freeze the evaluated implementation'):
        evaluate_confirmation(contract, path)


def test_final_partial_recording_resume_does_not_repeat_completed_predictions(trained_final_fixture, monkeypatch):
    import src.evaluation.m3w_confirmation_evaluation as module
    contract, _, _ = trained_final_fixture
    claim, cache = reserve(contract), {}

    def stop(key, rows):
        cache[key] = rows
        raise InterruptedError('Stopped after completed first recording')

    with pytest.raises(InterruptedError):
        evaluate_confirmation(contract, claim, on_recording=stop)
    assert set(cache) == {('s1', 't')}
    first_hash = contract.artifacts['s1_fit_ab']['sha256']
    original = module.decide_scene

    def decide(scene, model, *args, **kwargs):
        assert model._verified_artifact_sha256 != first_hash
        return original(scene, model, *args, **kwargs)

    monkeypatch.setattr(module, 'decide_scene', decide)
    report, rows = evaluate_confirmation(contract, claim, cached_rows=cache)
    assert rows['s1'] == cache['s1', 't'] and len(report['per_seed']) == 3


def test_changed_calibration_result_refused_before_final_labels(trained_final_fixture, monkeypatch):
    contract, _, _ = trained_final_fixture
    path = contract.root / 'cal/calibration_report.json'
    path.write_text(path.read_text() + ' ')
    monkeypatch.setattr(contract, 'open_recording', lambda *_a, **_k: pytest.fail('Final labels opened after calibration changed'))
    with pytest.raises(ValueError, match='artifact identity'):
        load_family(contract, device='cpu')


def test_final_cli_complete_resume_recovery_and_cache_tamper(trained_final_fixture):
    contract, paths, common = trained_final_fixture
    output = contract.root / 'final'
    command = [sys.executable, str(ROOT / 'scripts/evaluate_m3w_confirmation.py'), *common,
               '--artifacts', *map(str, paths), '--output-dir', str(output)]
    first = subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
    assert json.loads(first.stdout)['status'] == 'final_family_evaluated'
    digest = file_digest(output / 'confirmation_report.json')
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert repeated.returncode != 0
    resumed = subprocess.run(command + ['--resume'], check=True, capture_output=True, text=True, timeout=90)
    assert json.loads(resumed.stdout)['status'] == 'cached_verified'
    assert file_digest(output / 'confirmation_report.json') == digest
    (output / 'completion.json').unlink()
    recovered = subprocess.run(command + ['--resume'], check=True, capture_output=True, text=True, timeout=90)
    assert json.loads(recovered.stdout)['status'] == 'cached_verified_completion_recovered'
    cache = next(output.glob('*.rows.json')); cache.write_text(cache.read_text() + ' ')
    bad = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=90)
    assert bad.returncode != 0 and 'cache identity changed' in bad.stderr


def test_real_unapproved_protocol_is_not_opened():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_m3w_confirmation.py'), '--preflight-only'],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert json.loads(result.stdout) == {'status': 'refused', 'reason': 'Explicit protocol approval required', 'confirmation_evaluated': False}
