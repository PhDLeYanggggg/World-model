import numpy as np
import pytest

from src.data_unification.m3w_causal_recordings import write_recording
from src.evaluation.m3w_experiment_contract import (
    ContractError, ExperimentContract, protocol_digest, file_digest,
    claim_confirmation, finish_confirmation,
    claim_calibration, finish_calibration,
)


def fixture_contract(tmp_path):
    records, assignments = {}, {}
    for i, (name, role) in enumerate((('a', 'fit'), ('b', 'fit'), ('d', 'development'),
                                     ('c', 'calibration'), ('t', 'confirmation'))):
        directory = tmp_path / name
        points = np.array([[j*10, 1, j*.1+i, i] for j in range(25)])
        write_recording(directory, points, {'id': name, 'physical_scene': name})
        records[name] = {'physical_scene': name, 'cache_path': name,
                         'metadata_sha256': file_digest(directory / 'metadata.json'),
                         'historical_use': 'unexposed_reviewed',
                         'review_evidence': ['synthetic_fixture_not_real_authorization']}
        assignments[name] = role
    protocol = {
        'schema_version': 1, 'status': 'approved', 'scope': 'confirmatory',
        'confirmation_receipt': 'confirmation.json',
        'calibration_receipt': 'calibration.json',
        'records': records, 'assignments': assignments, 'fit_folds': {'a': 0, 'b': 1},
        'task': {'history_steps': 8, 'prediction_unit': 'observation_steps', 'horizon': 12,
                 'coordinate_claim': 'dataset_local_unverified',
                 'primary_metric': 'ade', 'aggregation': 'equal_physical_scene'},
        'risk': {'delta': .05, 'risks': [{'name': 'synthetic_bounded_loss',
                    'lower': 0., 'upper': 1., 'tolerance': .1}],
                 'easy_definition': 'synthetic_fixture', 'easy_degradation_max': .02},
        'seeds': [1, 2, 3], 'bootstrap_unit': 'physical_scene', 'bootstrap_resamples': 2000,
        'approval': None,
    }
    approve(protocol)
    return protocol


def approve(protocol):
    protocol['approval'] = {'approved_by': 'test_fixture_only',
                            'decision_reference': 'synthetic_unit_test_not_user_decision',
                            'protocol_sha256': protocol_digest(protocol)}


def development_only_protocol(tmp_path):
    p = fixture_contract(tmp_path)
    p.update(scope='exploratory', study_design='development_only')
    p['assignments'].update(c='excluded', t='excluded')
    p['risk'].update(delta=None, risks=[])
    approve(p)
    return p


def test_development_only_can_fit_without_invented_confirmation(tmp_path):
    p = development_only_protocol(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    assert not contract.confirmation_eligible_by_declaration
    assert len(contract.open_recording('a', purpose='fit')[1]) == 6
    assert len(contract.open_recording('d', purpose='development')[1]) == 6
    for stage, claim in [('calibration', claim_calibration), ('confirmation', claim_confirmation)]:
        with pytest.raises(ContractError, match='Development-only'):
            claim(tmp_path / f'{stage}.json', contract, ['model'])
        assert not (tmp_path / f'{stage}.json').exists()


@pytest.mark.parametrize('change', ['scope', 'reserved_role', 'missing_fit',
                                    'missing_development', 'formal_risk', 'unknown_design'])
def test_development_only_cannot_silently_become_confirmatory(tmp_path, change):
    p = development_only_protocol(tmp_path)
    if change == 'scope':
        p['scope'] = 'confirmatory'
    elif change == 'reserved_role':
        p['assignments']['c'] = 'calibration'
    elif change == 'missing_fit':
        p['assignments'].update(a='excluded', b='excluded')
    elif change == 'missing_development':
        p['assignments']['d'] = 'excluded'
    elif change == 'formal_risk':
        p['risk']['delta'] = .05
    else:
        p['study_design'] = 'arbitrary'
    approve(p)
    with pytest.raises(ContractError):
        ExperimentContract(p, tmp_path)


def artifact(tmp_path, protocol, name, *, kind='forecaster', fit=('a',), selected=(), calibrated=(), parents=()):
    path = tmp_path / f'{name}.bin'
    path.write_bytes(name.encode())
    return {'id': name, 'kind': kind, 'path': path.name, 'sha256': file_digest(path),
            'protocol_sha256': protocol_digest(protocol), 'lineage_complete': True,
            'fit_recordings': list(fit), 'selection_recordings': list(selected),
            'calibration_recordings': list(calibrated), 'parents': list(parents)}


def test_draft_and_unapproved_protocol_cannot_open_data(tmp_path):
    p = fixture_contract(tmp_path)
    p['status'], p['approval'] = 'draft', None
    with pytest.raises(ContractError, match='approval'):
        ExperimentContract(p, tmp_path)


def test_approval_binds_all_scientific_choices(tmp_path):
    p = fixture_contract(tmp_path)
    p['task']['primary_metric'] = 'fde'
    with pytest.raises(ContractError, match='digest'):
        ExperimentContract(p, tmp_path)


def test_distinct_recordings_in_one_scene_cannot_cross_roles(tmp_path):
    p = fixture_contract(tmp_path)
    p['records']['d']['physical_scene'] = 'a'
    approve(p)
    with pytest.raises(ContractError, match='physical scene'):
        ExperimentContract(p, tmp_path)


def test_history_exposure_cannot_be_renamed_into_confirmation_or_calibration(tmp_path):
    for name in ('c', 't'):
        p = fixture_contract(tmp_path)
        p['records'][name]['historical_use'] = 'development_exposed'
        approve(p)
        with pytest.raises(ContractError, match='historical'):
            ExperimentContract(p, tmp_path)
        p['scope'] = 'exploratory'
        approve(p)
        contract = ExperimentContract(p, tmp_path)
        assert not contract.confirmation_eligible_by_declaration


def test_unreviewed_exposure_is_not_confirmatory(tmp_path):
    p = fixture_contract(tmp_path)
    p['records']['t']['historical_use'] = 'unknown'
    approve(p)
    with pytest.raises(ContractError, match='historical'):
        ExperimentContract(p, tmp_path)


def test_folds_and_seeds_are_explicit_and_role_complete(tmp_path):
    p = fixture_contract(tmp_path)
    p['fit_folds']['d'] = 2
    approve(p)
    with pytest.raises(ContractError, match='fold'):
        ExperimentContract(p, tmp_path)
    p = fixture_contract(tmp_path)
    p['seeds'] = [1, 1, 1]
    approve(p)
    with pytest.raises(ContractError, match='seed'):
        ExperimentContract(p, tmp_path)


def test_hash_bound_reader_filters_exact_protocol_and_role(tmp_path):
    p = fixture_contract(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    reader, ids = contract.open_recording('a', purpose='fit')
    assert len(ids) == 6
    assert np.all(reader.index[ids]['protocol'] == 0)
    assert np.all(reader.index[ids]['future_steps'] == 12)
    with pytest.raises(ContractError, match='role'):
        contract.open_recording('t', purpose='fit')
    with pytest.raises(ContractError, match='confirmation claim'):
        contract.open_recording('t', purpose='confirmation')


def test_metadata_and_array_changes_invalidate_reader(tmp_path):
    p = fixture_contract(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    metadata = tmp_path / 'a/metadata.json'
    metadata.write_text(metadata.read_text() + ' ')
    with pytest.raises(ContractError, match='metadata'):
        contract.open_recording('a', purpose='fit')
    p = fixture_contract(tmp_path)
    contract = ExperimentContract(p, tmp_path)
    with (tmp_path / 'a/points.npy').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='identity'):
        contract.open_recording('a', purpose='fit')


def test_exact_raw25_is_rejected_not_substituted(tmp_path):
    p = fixture_contract(tmp_path)
    p['task']['prediction_unit'], p['task']['horizon'] = 'raw_annotation_frames', 25
    approve(p)
    contract = ExperimentContract(p, tmp_path)
    with pytest.raises(ContractError, match='No exact'):
        contract.open_recording('a', purpose='fit')


def test_parent_teacher_exposure_blocks_oof_even_if_child_is_clean(tmp_path):
    p = fixture_contract(tmp_path)
    teacher = artifact(tmp_path, p, 'teacher', fit=('a', 'b'))
    child = artifact(tmp_path, p, 'child', fit=('a',), parents=('teacher',))
    contract = ExperimentContract(p, tmp_path, [teacher, child])
    with pytest.raises(ContractError, match='exposure'):
        contract.assert_prediction_use('child', ['b'], purpose='oof_risk_training')


def test_clean_cross_fitted_teacher_and_all_parents_must_be_hashed(tmp_path):
    p = fixture_contract(tmp_path)
    teacher = artifact(tmp_path, p, 'teacher', fit=('a',))
    contract = ExperimentContract(p, tmp_path, [teacher])
    contract.assert_prediction_use('teacher', ['b'], purpose='oof_risk_training')
    (tmp_path / 'teacher.bin').write_bytes(b'new_weights')
    with pytest.raises(ContractError, match='artifact identity'):
        contract.assert_prediction_use('teacher', ['b'], purpose='oof_risk_training')


def test_test_exposed_or_missing_parent_model_is_not_legal(tmp_path):
    p = fixture_contract(tmp_path)
    for record in (artifact(tmp_path, p, 'bad', fit=('t',)),
                   artifact(tmp_path, p, 'missing', parents=('not_declared',))):
        with pytest.raises(ContractError):
            ExperimentContract(p, tmp_path, [record])


def test_calibration_does_not_refit_predictor_or_reuse_calibrated_policy(tmp_path):
    p = fixture_contract(tmp_path)
    bad = artifact(tmp_path, p, 'bad', kind='forecaster', calibrated=('c',))
    with pytest.raises(ContractError, match='calibrator'):
        ExperimentContract(p, tmp_path, [bad])
    model = artifact(tmp_path, p, 'model', fit=('a', 'b'), selected=('d',))
    gate = artifact(tmp_path, p, 'gate', kind='calibrator', fit=(), calibrated=('c',), parents=('model',))
    contract = ExperimentContract(p, tmp_path, [model, gate])
    contract.assert_prediction_use('model', ['c'], purpose='calibration')
    with pytest.raises(ContractError, match='exposure'):
        contract.assert_prediction_use('gate', ['c'], purpose='calibration')


def test_artifact_cycle_and_conflicting_same_bytes_lineage_rejected(tmp_path):
    p = fixture_contract(tmp_path)
    a = artifact(tmp_path, p, 'a_model', parents=('b_model',))
    b = artifact(tmp_path, p, 'b_model', parents=('a_model',))
    with pytest.raises(ContractError, match='cycle'):
        ExperimentContract(p, tmp_path, [a, b])
    a['parents'], b['parents'] = [], []
    b['path'], b['sha256'], b['fit_recordings'] = a['path'], a['sha256'], ['b']
    with pytest.raises(ContractError, match='conflicting'):
        ExperimentContract(p, tmp_path, [a, b])


def test_confirmation_is_claimed_once_and_resume_requires_frozen_identity(tmp_path):
    p = fixture_contract(tmp_path)
    model = artifact(tmp_path, p, 'model', fit=('a', 'b'), selected=('d',))
    contract = ExperimentContract(p, tmp_path, [model])
    claim = tmp_path / 'confirmation.json'
    claim_confirmation(claim, contract, ['model'])
    reader, ids = contract.open_recording('t', purpose='confirmation', claim_path=claim)
    assert len(ids) and reader.metadata['id'] == 't'
    with pytest.raises(ContractError, match='already'):
        claim_confirmation(claim, contract, ['model'])
    claim_confirmation(claim, contract, ['model'], resume=True)
    other = artifact(tmp_path, p, 'other', fit=('a', 'b'))
    changed = ExperimentContract(p, tmp_path, [other])
    with pytest.raises(ContractError, match='identity'):
        claim_confirmation(claim, changed, ['other'], resume=True)
    results = tmp_path / 'result.json'
    results.write_text('{"fixture_only": true}')
    finish_confirmation(claim, contract, ['model'], results)
    with pytest.raises(ContractError, match='completed'):
        claim_confirmation(claim, contract, ['model'], resume=True)
    with pytest.raises(ContractError, match='completed'):
        contract.open_recording('t', purpose='confirmation', claim_path=claim)


def test_claim_rejects_forecast_model_changed_after_start(tmp_path):
    p = fixture_contract(tmp_path)
    model = artifact(tmp_path, p, 'model', fit=('a',))
    contract = ExperimentContract(p, tmp_path, [model])
    claim = tmp_path / 'confirmation.json'
    claim_confirmation(claim, contract, ['model'])
    (tmp_path / 'model.bin').write_bytes(b'mutated')
    with pytest.raises(ContractError, match='artifact identity'):
        contract.open_recording('t', purpose='confirmation', claim_path=claim)


def test_same_physical_scene_cannot_cross_crossfit_folds(tmp_path):
    p = fixture_contract(tmp_path)
    p['records']['b']['physical_scene'] = 'a'
    approve(p)
    with pytest.raises(ContractError, match='fold'):
        ExperimentContract(p, tmp_path)


def test_unknown_role_and_unbounded_risk_are_not_silently_defaulted(tmp_path):
    for mutate in (lambda p: p['assignments'].update(t='test'),
                   lambda p: p['risk']['risks'][0].update(upper=float('inf'))):
        p = fixture_contract(tmp_path)
        mutate(p)
        with pytest.raises((ContractError, ValueError)):
            approve(p)
            ExperimentContract(p, tmp_path)


def test_second_receipt_path_cannot_bypass_one_confirmation(tmp_path):
    p = fixture_contract(tmp_path)
    model = artifact(tmp_path, p, 'model')
    contract = ExperimentContract(p, tmp_path, [model])
    with pytest.raises(ContractError, match='receipt path'):
        claim_confirmation(tmp_path / 'another_receipt.json', contract, ['model'])
    assert not (tmp_path / 'another_receipt.json').exists()


def test_mutated_in_memory_protocol_or_lineage_invalidates_access(tmp_path):
    p = fixture_contract(tmp_path)
    model = artifact(tmp_path, p, 'model')
    contract = ExperimentContract(p, tmp_path, [model])
    contract.protocol['task']['horizon'] = 50
    with pytest.raises(ContractError, match='identity changed'):
        contract.open_recording('a', purpose='fit')
    contract = ExperimentContract(p, tmp_path, [model])
    contract.artifacts['model']['fit_recordings'] = []
    with pytest.raises(ContractError, match='identity changed'):
        contract.assert_prediction_use('model', ['b'], purpose='oof_risk_training')


def test_oof_excludes_the_entire_declared_fold_not_just_target_record(tmp_path):
    p = fixture_contract(tmp_path)
    points = np.array([[j*10, 1, j*.1+10, 10] for j in range(25)])
    write_recording(tmp_path / 'a2', points, {'id': 'a2', 'physical_scene': 'another_scene'})
    p['records']['a2'] = {**p['records']['a'], 'physical_scene': 'another_scene', 'cache_path': 'a2',
                          'metadata_sha256': file_digest(tmp_path / 'a2/metadata.json')}
    p['assignments']['a2'], p['fit_folds']['a2'] = 'fit', 0
    approve(p)
    model = artifact(tmp_path, p, 'model', fit=('a2',))
    contract = ExperimentContract(p, tmp_path, [model])
    with pytest.raises(ContractError, match='exposure'):
        contract.assert_prediction_use('model', ['a'], purpose='oof_risk_training')


def test_identical_arrays_under_renamed_scenes_are_rejected(tmp_path):
    p = fixture_contract(tmp_path)
    points = np.array([[j*10, 1, j*.1, 0] for j in range(25)])
    write_recording(tmp_path / 't', points, {'id': 't', 'physical_scene': 't'})
    p['records']['t']['metadata_sha256'] = file_digest(tmp_path / 't/metadata.json')
    approve(p)
    with pytest.raises(ContractError, match='Byte-identical'):
        ExperimentContract(p, tmp_path)


def test_code_binding_changes_invalidate_existing_contract(tmp_path):
    p = fixture_contract(tmp_path)
    code = tmp_path / 'reader.py'
    code.write_text('# fixture v1\n')
    p['bindings'] = {'reader.py': file_digest(code)}
    approve(p)
    contract = ExperimentContract(p, tmp_path)
    code.write_text('# fixture v2\n')
    with pytest.raises(ContractError, match='binding'):
        contract.open_recording('a', purpose='fit')


def test_calibration_family_freezes_before_access_and_cannot_grow_after_peeking(tmp_path):
    p = fixture_contract(tmp_path)
    model = artifact(tmp_path, p, 'model')
    other = artifact(tmp_path, p, 'other')
    contract = ExperimentContract(p, tmp_path, [model, other])
    with pytest.raises(ContractError, match='calibration claim'):
        contract.open_recording('c', purpose='calibration')
    path = tmp_path / 'calibration.json'
    claim_calibration(path, contract, ['model'])
    _, rows = contract.open_recording('c', purpose='calibration', claim_path=path)
    assert len(rows)
    with pytest.raises(ContractError, match='identity'):
        claim_calibration(path, contract, ['model', 'other'], resume=True)
    result = tmp_path / 'calibrated.json'
    result.write_text('{"fixture": true}')
    finish_calibration(path, contract, ['model'], result)
    with pytest.raises(ContractError, match='completed'):
        contract.open_recording('c', purpose='calibration', claim_path=path)


def test_same_receipt_path_alias_cannot_merge_calibration_with_confirmation(tmp_path):
    p = fixture_contract(tmp_path)
    p['calibration_receipt'] = './confirmation.json'
    approve(p)
    with pytest.raises(ContractError, match='Separate'):
        ExperimentContract(p, tmp_path)
