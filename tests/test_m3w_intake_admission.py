import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_m3w_experiment_contract import fixture_contract, approve
from src.evaluation.m3w_experiment_contract import ContractError, ExperimentContract, file_digest


def mark_pending(protocol, root, name='a'):
    path = root / name / 'metadata.json'
    content = json.loads(path.read_text())
    content['source_conditions_review'] = 'pending_not_formal_use_approval'
    content['historical_predictive_use'] = 'unknown_not_proven_untouched'
    path.write_text(json.dumps(content))
    protocol['records'][name]['metadata_sha256'] = file_digest(path)
    approve(protocol)


def test_approved_scientific_roles_cannot_promote_pending_intake_without_review(tmp_path):
    protocol = fixture_contract(tmp_path)
    mark_pending(protocol, tmp_path)
    with pytest.raises(ContractError, match='intake'):
        ExperimentContract(protocol, tmp_path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return {'path': path.name, 'sha256': file_digest(path)}


def intake_fixture(root, *, name='a', quarantined=False, source_approved=True):
    protocol = fixture_contract(root)
    mark_pending(protocol, root, name)
    metadata_path = root / name / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    source_path = root / 'raw.txt'
    source_path.write_text('synthetic source identity fixture only')
    files = [{'path': 'raw.txt', 'sha256': file_digest(source_path)}]
    metadata.update(dataset='SYNTHETIC_INTAKE', files=files)
    metadata_path.write_text(json.dumps(metadata))
    protocol['records'][name]['metadata_sha256'] = file_digest(metadata_path)
    source = write_json(root / 'source.json', {'status': 'verified', 'upstream_commit': 'synthetic_fixture', 'files': files})
    conversion = write_json(root / 'conversion.json', {'upstream_commit': 'synthetic_fixture', 'recordings': [
        {'id':name, 'points':metadata['points'], 'physical_scene':name}]})
    flags = [{'recording':name, 'issue':'synthetic_duplicate_agent'}] if quarantined else []
    audit = write_json(root / 'audit.json', {'source_manifest_sha256':source['sha256'],
        'build_report_sha256':conversion['sha256'], 'quarantine_recordings':[name] if quarantined else [],
        'annotation_flags':flags, 'clips':[{'recording':name, 'physical_scene':name,'points':metadata['points']}]})
    entry = {'cache_path':name, 'metadata_sha256':file_digest(metadata_path), 'physical_scene':name,
        'dataset':'SYNTHETIC_INTAKE', 'source_file_hashes':{'raw.txt':file_digest(source_path)},
        'quality_disposition':'quarantine' if quarantined else 'no_flags_in_limited_screen', 'blocking_findings':flags}
    screen = {'schema_version':1, 'kind':'m3w_diagnostic_intake_screen', 'source_manifest':source,
              'conversion_report':conversion, 'quality_audit':audit, 'recordings':{name:entry}}
    protocol['records'][name]['intake_screen'] = write_json(root / 'screen.json', screen)
    if source_approved:
        evidence = write_json(root / 'source_use.json', {'synthetic_test_only':True, 'not_real_approval':True})
        protocol['records'][name]['source_use_decision'] = {'status':'reviewed_for_declared_roles',
            'reviewed_by':'synthetic_fixture_only', 'decision_reference':'unit_test_not_user_or_source_authorization',
            'allowed_roles':[protocol['assignments'][name]], 'evidence':[evidence]}
    approve(protocol)
    return protocol


@pytest.mark.parametrize('name', ['a','d','c','t'])
def test_no_source_use_decision_blocks_each_role_before_reader(tmp_path, monkeypatch, name):
    import src.evaluation.m3w_experiment_contract as module
    p = intake_fixture(tmp_path, name=name, source_approved=False)
    def must_not_open(*args, **kwargs):
        raise AssertionError('Recording arrays were opened before admission')
    monkeypatch.setattr(module, 'RecordingWindows', must_not_open)
    with pytest.raises(ContractError, match='source-use review'):
        ExperimentContract(p, tmp_path)


@pytest.mark.parametrize('name', ['a','d','c','t'])
def test_approved_source_use_does_not_override_quality_quarantine(tmp_path, name):
    p = intake_fixture(tmp_path, name=name, quarantined=True)
    with pytest.raises(ContractError, match='quality quarantine'):
        ExperimentContract(p, tmp_path)


def test_clear_synthetic_review_can_train_with_same_causal_reader(tmp_path):
    p = intake_fixture(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    reader, ids = contract.open_recording('a', purpose='fit')
    assert len(ids) == 6 and reader.metadata['id'] == 'a'
    assert set(contract._intake_bindings) == {'screen.json','source.json','conversion.json','audit.json','source_use.json'}


@pytest.mark.parametrize('file', ['screen.json','source.json','conversion.json','audit.json','source_use.json'])
def test_evidence_drift_rejected_at_start_and_after_construction(tmp_path, file):
    p = intake_fixture(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    path = tmp_path / file
    path.write_text(path.read_text()+' ')
    with pytest.raises(ContractError, match='intake evidence'):
        ExperimentContract(p, tmp_path)
    with pytest.raises(ContractError, match='intake evidence'):
        contract.open_recording('a', purpose='fit')


def test_clearing_summary_flag_cannot_hide_underlying_audit(tmp_path):
    p = intake_fixture(tmp_path, quarantined=True)
    path = tmp_path / 'screen.json'
    screen = json.loads(path.read_text())
    screen['recordings']['a'].update(quality_disposition='no_flags_in_limited_screen', blocking_findings=[])
    p['records']['a']['intake_screen'] = write_json(path, screen)
    approve(p)
    with pytest.raises(ContractError, match='quality quarantine'):
        ExperimentContract(p, tmp_path)


@pytest.mark.parametrize('change', ['identity','files','points','missing_quarantine','bad_source','cross_audit'])
def test_false_rehashed_screen_or_audit_is_rejected(tmp_path, change):
    p = intake_fixture(tmp_path)
    path = tmp_path / 'screen.json'
    screen = json.loads(path.read_text())
    if change == 'identity':
        screen['recordings']['a']['metadata_sha256'] = '0'*64
    elif change == 'files':
        screen['recordings']['a']['source_file_hashes'] = {'unverified':'0'*64}
    else:
        audit_path = tmp_path / 'audit.json'
        audit = json.loads(audit_path.read_text())
        if change == 'points':
            audit['clips'][0]['points'] += 1
        elif change == 'missing_quarantine':
            del audit['quarantine_recordings']
        elif change == 'bad_source':
            source_path = tmp_path / 'source.json'
            source = json.loads(source_path.read_text())
            source['status'] = 'not_verified'
            screen['source_manifest'] = write_json(source_path, source)
            audit['source_manifest_sha256'] = screen['source_manifest']['sha256']
        else:
            audit['source_manifest_sha256'] = '0'*64
        screen['quality_audit'] = write_json(audit_path, audit)
    p['records']['a']['intake_screen'] = write_json(path, screen)
    approve(p)
    with pytest.raises(ContractError, match='Intake admission refused'):
        ExperimentContract(p, tmp_path)


@pytest.mark.parametrize('change', ['role','evidence_missing','decision_missing','text_instead_of_decision','escape','bad_hash'])
def test_source_use_declaration_must_be_scoped_and_hash_bound(tmp_path, change):
    p = intake_fixture(tmp_path)
    decision = p['records']['a']['source_use_decision']
    if change == 'role':
        decision['allowed_roles'] = ['development']
    elif change == 'evidence_missing':
        decision['evidence'] = []
    elif change == 'decision_missing':
        decision['decision_reference'] = ''
    elif change == 'text_instead_of_decision':
        p['records']['a']['source_use_decision'] = 'pending_not_issued_by_this_script'
    elif change == 'escape':
        decision['evidence'][0]['path'] = '/etc/hosts'
    else:
        decision['evidence'][0]['sha256'] = 'wrong_hash'
    approve(p)
    with pytest.raises(ContractError):
        ExperimentContract(p, tmp_path)


def test_intake_fields_cannot_change_after_scientific_approval(tmp_path):
    p = intake_fixture(tmp_path)
    p['records']['a']['source_use_decision']['allowed_roles'].append('confirmation')
    with pytest.raises(ContractError, match='digest changed'):
        ExperimentContract(p, tmp_path)


def test_excluded_source_remains_in_catalog_without_becoming_accessible(tmp_path):
    p = fixture_contract(tmp_path)
    # The quarantined object can remain in the inventory; no usable role is granted.
    p['records']['q'] = dict(p['records']['a'])
    content = json.loads((tmp_path / 'a/metadata.json').read_text())
    content.update(id='q', physical_scene='q', source_conditions_review='pending')
    path = tmp_path / 'q/metadata.json'
    path.parent.mkdir()
    path.write_text(json.dumps(content))
    p['records']['q'].update(cache_path='q', physical_scene='q', metadata_sha256=file_digest(path))
    p['assignments']['q'] = 'excluded'
    approve(p)
    contract = ExperimentContract(p, tmp_path)
    with pytest.raises(ContractError, match='data role'):
        contract.open_recording('q', purpose='fit')


def test_production_training_cli_refuses_quarantine_before_torch_or_checkpoint(tmp_path):
    p = intake_fixture(tmp_path, quarantined=True)
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(p))
    output = tmp_path / 'must_not_train'
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / 'scripts/train_m3w_causal_forecaster.py'),
        '--protocol', str(protocol_path), '--workspace-root', str(tmp_path), '--output-dir', str(output),
        '--fit-recordings', 'a', '--baseline', 'constant_velocity_causal_fd', '--seed', '1'],
        cwd=repo, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    message = json.loads(result.stdout)
    assert not message['torch_training_started']
    assert 'quality quarantine' in message['reason']
    assert not output.exists()


@pytest.mark.parametrize('value', [[], 'not_a_recording_table', None])
def test_malformed_screen_refuses_with_contract_error(tmp_path, value):
    p = intake_fixture(tmp_path)
    path = tmp_path / 'screen.json'
    screen = json.loads(path.read_text())
    screen['recordings'] = value
    p['records']['a']['intake_screen'] = write_json(path, screen)
    approve(p)
    with pytest.raises(ContractError, match='recording table'):
        ExperimentContract(p, tmp_path)


def test_admitted_synthetic_fit_runs_torch_checkpoint_and_resume(tmp_path):
    import torch
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, train_forecaster
    p = intake_fixture(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    data = ContractForecastDataset(contract, ['a'], purpose='fit', baseline_name='constant_velocity_causal_fd')
    architecture = {'width':8, 'heads':2, 'layers':1}
    settings = {'seed':1, 'steps':4, 'batch_size':2, 'learning_rate':.003,
                'checkpoint_every':2, 'heartbeat_every':1}
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        partial = train_forecaster(data, architecture=architecture, settings=settings, output_dir=tmp_path/'fit', stop_after=2)
        assert partial['steps_completed'] == 2 and not partial['training_complete']
        before = torch.load(partial['checkpoint'], map_location='cpu', weights_only=True)['model']
        complete = train_forecaster(data, architecture=architecture, settings=settings, output_dir=tmp_path/'fit', resume=True)
        after = torch.load(complete['checkpoint'], map_location='cpu', weights_only=True)['model']
        assert complete['training_complete'] and complete['resumed_from_step'] == 2
        assert any(not torch.equal(before[k], after[k]) for k in before)
        assert all(torch.isfinite(torch.tensor(complete['losses'])))
        reused = train_forecaster(data, architecture=architecture, settings=settings, output_dir=tmp_path/'fit', resume=True)
        assert reused['result_source'] == 'cached_verified'
        assert len((tmp_path/'fit/heartbeat.jsonl').read_text().splitlines()) == 4
        # A source decision cannot be revoked/changed and then silently resumed.
        (tmp_path/'source_use.json').write_text('{"changed":true}')
        with pytest.raises(ContractError, match='intake evidence'):
            train_forecaster(data, architecture=architecture, settings=settings, output_dir=tmp_path/'fit', resume=True)
    finally:
        torch.set_num_threads(previous)
