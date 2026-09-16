from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_m3w_development_evaluation import fitted_fixture, GEOMETRY, POLICY
from test_m3w_experiment_contract import fixture_contract, approve
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, claim_calibration
from src.evaluation.m3w_risk_calibration import (
    validate_rules, load_frozen_policy, decide_frozen, aggregate_bounded_risks, screen_policies, evaluate_calibration,
)
from src.evaluation.m3w_development_evaluation import content_digest

ROOT = Path(__file__).resolve().parents[1]
RISKS = [{'name': 'bounded_harm', 'statistic': 'clipped_positive_excess', 'lower': 0., 'upper': 1.,
          'tolerance': .02, 'clip_scale': 1.},
         {'name': 'harm_probability', 'statistic': 'harm_event', 'lower': 0., 'upper': 1.,
          'tolerance': .1, 'margin': .05}]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def add_rules(p):
    p['risk']['risks'] = deepcopy(RISKS)
    p['calibration_evaluation'] = {
        'error_unit': 'past_normalized', 'label_policy': 'complete_requested_path',
        'query_stride': 2, 'geometry_by_recording': {'c': GEOMETRY}, 'solver_seconds': 2.,
        'within_scene_aggregation': 'agent_window', 'selection_rule': 'first_accepted_in_frozen_development_order',
        'policy_priority': ['dev'],
    }


def calibrated_fixture(tmp_path, *, learned=False):
    def fixture_rules(p):
        add_rules(p)
        if learned:
            # This integration fixture needs nonempty synthetic easy support.
            # These values never alter the real, still-unapproved protocol.
            p['development_evaluation']['easy_threshold'] = 1.
            p['development_evaluation']['hard_threshold'] = 2.
            p['risk']['easy_definition']['threshold'] = 1.

    kwargs = {'baseline': 'constant_position', 'steps': 80} if learned else {}
    contract, plan = fitted_fixture(tmp_path, protocol_updates=fixture_rules, **kwargs)
    ppath, planpath = tmp_path / 'protocol.json', tmp_path / 'plan.json'
    ppath.write_text(json.dumps(contract.protocol))
    planpath.write_text(json.dumps(plan))
    paths = []
    for name, record in contract.artifacts.items():
        path = tmp_path / (name + '.artifact.json')
        path.write_text(json.dumps(record))
        paths.append(path)
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_m3w_development.py'), '--workspace-root', str(tmp_path),
        '--protocol', str(ppath), '--plan', str(planpath), '--artifacts', *map(str, paths),
        '--output-dir', str(tmp_path / 'dev'), '--threads', '2'], check=True, capture_output=True, timeout=45)
    selected = json.loads((tmp_path / 'dev/artifact.json').read_text())
    return ExperimentContract(contract.protocol, tmp_path, [*contract.artifacts.values(), selected]), paths


def row(recording, scene, agent, baseline, prediction, *, switch=True, name='p', frame=80):
    return {'recording_id': recording, 'physical_scene': scene, 'frame_id': frame, 'agent_id': agent,
            'horizon_raw': 120, 'requested_steps': 12, 'scale': 1., 'baseline_ade': baseline,
            'arms': {name: {'ade': prediction, 'switch': switch}}}


def test_unknown_switched_labels_receive_worst_risk_and_no_switch_exact_zero():
    rows = [row('c', 'scene', 1, None, None), row('c', 'scene', 2, None, None, switch=False),
            row('c', 'scene', 3, .1, .3), row('c', 'scene', 4, .2, .1)]
    scenes, risk, coverage = aggregate_bounded_risks({'p': rows}, [{'id': 'p', 'arm': 'joint'}],
        records={'c': {'physical_scene': 'scene'}}, risks=RISKS, metric='ade', within_scene_aggregation='agent_window')
    assert scenes == ['scene']
    np.testing.assert_allclose(risk[0, 0], [.3, .5])
    assert coverage['p']['past_supported_decisions'] == 4
    assert coverage['p']['observed_primary_labels'] == 2 and coverage['p']['unknown_switched_worst_case'] == 1


def test_scene_aggregation_is_not_window_replication_as_independent_units():
    rows = [row('c1', 's', 1, 0., 1.)] + [row('c2', 's', i, 0., 0.) for i in range(1, 10)]
    records = {'c1': {'physical_scene': 's'}, 'c2': {'physical_scene': 's'}}
    args = dict(records=records, risks=RISKS, metric='ade')
    scenes, equal, _ = aggregate_bounded_risks({'p': rows}, [{'id': 'p', 'arm': 'joint'}], **args, within_scene_aggregation='equal_recording')
    _, pooled, _ = aggregate_bounded_risks({'p': rows}, [{'id': 'p', 'arm': 'joint'}], **args, within_scene_aggregation='agent_window')
    assert len(scenes) == 1
    np.testing.assert_allclose(equal[0, 0], [.5, .5])
    np.testing.assert_allclose(pooled[0, 0], [.1, .1])


@pytest.mark.parametrize('violation', ['duplicate', 'wrong_scene', 'wrong_policy', 'nonfinite', 'not_floor', 'baseline_changed'])
def test_invalid_risk_rows_are_refused(violation):
    rows = [row('c', 's', 1, .1, .2)]
    policies = [{'id': 'p', 'arm': 'joint'}]
    if violation == 'duplicate':
        rows *= 2
    elif violation == 'wrong_scene':
        rows[0]['physical_scene'] = 'other'
    elif violation == 'wrong_policy':
        rows[0]['arms'] = {'other': rows[0]['arms']['p']}
    elif violation == 'nonfinite':
        rows[0]['arms']['p']['ade'] = float('nan')
    elif violation == 'not_floor':
        policies[0]['arm'] = 'floor'
    else:
        rows[0]['arms']['p']['switch'] = False
    with pytest.raises(ValueError):
        aggregate_bounded_risks({'p': rows}, policies, records={'c': {'physical_scene': 's'}},
                               risks=RISKS, metric='ade', within_scene_aggregation='agent_window')


def test_policies_must_share_query_and_baseline_identity():
    a, b = row('c', 's', 1, .1, .2, name='p'), row('c', 's', 1, .2, .2, name='q')
    with pytest.raises(ValueError, match='share baseline'):
        aggregate_bounded_risks({'p': [a], 'q': [b]}, [{'id': 'p', 'arm': 'joint'}, {'id': 'q', 'arm': 'joint'}],
            records={'c': {'physical_scene': 's'}}, risks=RISKS, metric='ade', within_scene_aggregation='agent_window')


def test_zero_observed_harm_does_not_prove_learned_policy_safe_with_one_scene(tmp_path):
    p = fixture_contract(tmp_path)
    add_rules(p); approve(p)
    contract = ExperimentContract(p, tmp_path)
    rows = {'dev': [row('c', 'c', 1, .1, .1, switch=False, name='dev')]}
    learned = screen_policies(contract, [{'id': 'dev', 'arm': 'joint'}], rows)
    assert learned['accepted_under_assumptions'] == [False]
    assert learned['selected_policy_id'] is None and learned['use_unchanged_baseline']
    assert learned['upper_risk_bounds'][0][0] > .02
    exact = screen_policies(contract, [{'id': 'dev', 'arm': 'floor'}], rows)
    assert exact['upper_risk_bounds'] == [[0., 0.]] and exact['accepted_under_assumptions'] == [True]
    assert not exact['deployment_approved'] and not exact['raw_ADE_FDE_or_easy_relative_degradation_certified']


def test_risks_require_functional_and_priority_not_just_names(tmp_path):
    p = fixture_contract(tmp_path)
    add_rules(p); approve(p)
    c = ExperimentContract(p, tmp_path)
    assert validate_rules(c)[1] == ['c']
    p['risk']['risks'][0].pop('statistic'); approve(p)
    with pytest.raises(ValueError, match='risk functional'):
        validate_rules(ExperimentContract(p, tmp_path))
    add_rules(p)
    p['calibration_evaluation']['policy_priority'] = []; approve(p)
    with pytest.raises(ValueError, match='priority'):
        validate_rules(ExperimentContract(p, tmp_path))


def test_actual_development_export_calibrates_only_calibration_labels(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    import src.evaluation.m3w_risk_calibration as module
    contract, _ = calibrated_fixture(tmp_path)
    claim = tmp_path / 'calibration.json'
    claim_calibration(claim, contract, ['dev'])
    events = []
    original_open, original_decide, original_labels = contract.open_recording, module.decide_frozen, RecordingWindows.get_scene_labels

    def opened(name, **kwargs):
        assert name == 'c' and kwargs['purpose'] == 'calibration'
        events.append(('open', name))
        return original_open(name, **kwargs)

    def decided(scene, *args, **kwargs):
        value = original_decide(scene, *args, **kwargs)
        events.append(('decision', scene['frame_id']))
        return value

    def labeled(reader, scene):
        assert reader.metadata['id'] == 'c' and events[-1] == ('decision', scene['frame_id'])
        events.append(('labels', scene['frame_id']))
        return original_labels(reader, scene)

    monkeypatch.setattr(contract, 'open_recording', opened)
    monkeypatch.setattr(module, 'decide_frozen', decided)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', labeled)
    result, rows = evaluate_calibration(contract, ['dev'], claim)
    assert events and result['physical_scene_count_declared'] == 1
    assert result['use_unchanged_baseline']
    assert result['coverage']['dev']['past_supported_decisions'] > result['coverage']['dev']['observed_primary_labels']
    assert not result['independence_verified'] and not result['confirmation_evaluated']
    assert {r['recording_id'] for r in rows['dev']} == {'c'}


def test_calibration_without_claim_refused_before_readers(tmp_path, monkeypatch):
    contract, _ = calibrated_fixture(tmp_path)
    monkeypatch.setattr(contract, 'open_recording', lambda *_a, **_k: pytest.fail('Reader opened without frozen claim'))
    with pytest.raises(FileNotFoundError):
        evaluate_calibration(contract, ['dev'], tmp_path / 'calibration.json')


def test_changed_development_evidence_refused_before_calibration_labels(tmp_path, monkeypatch):
    contract, _ = calibrated_fixture(tmp_path)
    path = tmp_path / 'dev/development_report.json'
    path.write_text(path.read_text() + ' ')
    monkeypatch.setattr(contract, 'open_recording', lambda *_a, **_k: pytest.fail('Reader opened before integrity check'))
    claim_calibration(tmp_path / 'calibration.json', contract, ['dev'])
    with pytest.raises(ValueError, match='artifact changed'):
        evaluate_calibration(contract, ['dev'], tmp_path / 'calibration.json')


def test_policy_implementation_must_match_development_version(tmp_path):
    contract, _ = calibrated_fixture(tmp_path)
    path = tmp_path / 'dev/run_identity.json'
    identity = json.loads(path.read_text())
    identity['code_sha256']['src/world_model/m3w_joint_intervention.py'] = 'changed'
    path.write_text(json.dumps(identity))
    complete_path = tmp_path / 'dev/completion.json'
    complete = json.loads(complete_path.read_text())
    complete['run_sha256'] = content_digest(identity)
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='policy implementation changed'):
        load_frozen_policy(contract, 'dev', device='cpu')


def test_learned_policy_runs_frozen_models_but_small_calibration_cannot_certify(tmp_path):
    from src.world_model.m3w_supervised_intervention import parameter_digest
    contract, _ = calibrated_fixture(tmp_path, learned=True)
    policy = load_frozen_policy(contract, 'dev', device='cpu')
    assert policy['arm'] != 'floor' and policy['model'] is not None
    before = {name: file_digest(contract._path(record['path'])) for name, record in contract.artifacts.items()}
    weights = parameter_digest(policy['model'])
    claim_calibration(tmp_path / 'calibration.json', contract, ['dev'])
    report, _ = evaluate_calibration(contract, ['dev'], tmp_path / 'calibration.json')
    assert report['statistical_policy_family_size'] == 1
    assert report['coverage']['dev']['switches'] > 0
    assert report['accepted_under_assumptions'] == [False] and report['use_unchanged_baseline']
    assert parameter_digest(policy['model']) == weights
    assert before == {name: file_digest(contract._path(record['path'])) for name, record in contract.artifacts.items()}


def test_interrupted_recording_cache_reused_without_reopening_labels(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    contract, _ = calibrated_fixture(tmp_path)
    claim = tmp_path / 'calibration.json'
    claim_calibration(claim, contract, ['dev'])
    cache = {}

    def interrupted(key, rows):
        cache[key] = rows
        raise InterruptedError('Simulated stop after atomic recording save')

    with pytest.raises(InterruptedError):
        evaluate_calibration(contract, ['dev'], claim, on_recording=interrupted)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', lambda *_: pytest.fail('Completed cached recording labels reopened'))
    result, _ = evaluate_calibration(contract, ['dev'], claim, cached_rows=cache)
    assert result['coverage']['dev']['past_supported_decisions'] == len(cache['dev', 'c'])


def test_policy_order_is_protocol_bound_before_claim_or_label_access(tmp_path):
    p = fixture_contract(tmp_path)
    add_rules(p)
    p['calibration_evaluation']['policy_priority'] = ['p', 'q']
    approve(p)
    contract = ExperimentContract(p, tmp_path)
    with pytest.raises(ValueError, match='frozen calibration priority'):
        evaluate_calibration(contract, ['q', 'p'], tmp_path / 'calibration.json')


def test_support_calculation_and_pseudo_replication_diagnostic():
    from scripts.audit_m3w_risk_support import sensitivity, correlated_window_diagnostic
    table = sensitivity(6)
    assert len(table) == 12 and not any(r['any_accepted'] for r in table)
    assert table[0]['optimistic_zero_loss_scenes_required_for_this_bound'] == 3745
    assert table[0]['upper_bound'] == pytest.approx(np.sqrt(np.log(20) / 12))
    diagnostic = correlated_window_diagnostic()
    assert diagnostic['correct_cluster_false_acceptance_fraction'] == 0
    assert .25 < diagnostic['naive_window_false_acceptance_fraction'] < .28


def test_calibration_cli_resume_and_cache_tamper_are_enforced(tmp_path):
    contract, paths = calibrated_fixture(tmp_path)
    paths.append(tmp_path / 'dev/artifact.json')
    common = [sys.executable, str(ROOT / 'scripts/calibrate_m3w_intervention.py'), '--protocol', str(tmp_path / 'protocol.json'),
              '--workspace-root', str(tmp_path), '--artifacts', *map(str, paths), '--policy-ids', 'dev',
              '--output-dir', str(tmp_path / 'cal'), '--threads', '2']
    first = subprocess.run(common, check=True, capture_output=True, text=True, timeout=45)
    assert json.loads(first.stdout)['use_unchanged_baseline']
    assert json.loads((tmp_path / 'calibration.json').read_text())['state'] == 'completed'
    calibrator = json.loads((tmp_path / 'cal/artifact.json').read_text())
    assert calibrator['calibration_recordings'] == ['c'] and calibrator['parents'] == ['dev']
    extended = ExperimentContract(contract.protocol, tmp_path, [*contract.artifacts.values(), calibrator])
    extended.assert_prediction_use('cal', ['t'], purpose='confirmation')
    result = tmp_path / 'cal/calibration_report.json'
    digest = file_digest(result)
    resumed = subprocess.run(common + ['--resume'], check=True, capture_output=True, text=True, timeout=45)
    assert json.loads(resumed.stdout)['status'] == 'cached_verified' and file_digest(result) == digest
    (tmp_path / 'cal/completion.json').unlink()
    recovered = subprocess.run(common + ['--resume'], check=True, capture_output=True, text=True, timeout=45)
    assert json.loads(recovered.stdout)['status'] == 'cached_verified_completion_recovered'
    cache = next((tmp_path / 'cal').glob('*.rows.json'))
    cache.write_text(cache.read_text() + ' ')
    failure = subprocess.run(common + ['--resume'], capture_output=True, text=True, timeout=45)
    assert failure.returncode != 0 and 'cache identity changed' in failure.stderr
    assert not (tmp_path / 'confirmation.json').exists()


def test_real_draft_refuses_calibration_before_torch():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/calibrate_m3w_intervention.py'), '--preflight-only'],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert json.loads(result.stdout) == {'status': 'refused', 'reason': 'Explicit protocol approval required', 'calibration_evaluated': False}
