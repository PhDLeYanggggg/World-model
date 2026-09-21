import pytest

from src.data_unification.m3w_causal_recordings import write_recording
from src.evaluation.m3w_experiment_contract import ContractError, ExperimentContract, file_digest
from src.evaluation.m3w_cost_validation_lineage import (
    audit_cost_validation, audit_prediction_lineage, require_cost_validation,
)
from test_m3w_experiment_contract import approve, artifact, fixture_contract


def three_folds(tmp_path):
    import numpy as np
    p = fixture_contract(tmp_path)
    directory = tmp_path / 'e'
    write_recording(directory, np.array([[j*10, 1, j*.2, 9] for j in range(25)]),
                    {'id': 'e', 'physical_scene': 'e'})
    p['records']['e'] = dict(p['records']['a'], physical_scene='e', cache_path='e',
                             metadata_sha256=file_digest(directory / 'metadata.json'))
    p['assignments']['e'] = 'fit'
    p['fit_folds']['e'] = 2
    approve(p)
    return p


def test_oof_producer_is_clean_but_reused_training_groups_expose_outer_fold(tmp_path):
    p = three_folds(tmp_path)
    producers = [artifact(tmp_path, p, 'hold_'+r, fit=tuple(s for s in 'abe' if s != r))
                 for r in 'abe']
    c = ExperimentContract(p, tmp_path, producers)
    for r in 'abe':
        assert audit_prediction_lineage(c, 'hold_'+r, [r])['eligible_under_declared_lineage']
    groups = [{'recordings': [r], 'predictor_id': 'hold_'+r} for r in 'be']
    result = audit_cost_validation(c, ['a'], groups, validation_producer_id='hold_a')
    assert result['direct_training_scene_overlap'] == []
    assert result['upstream_training_scene_overlap'] == ['a']
    assert not result['eligible_under_declared_lineage']
    assert result['validation_producer']['eligible_under_declared_lineage']
    assert {v['artifact_id'] for v in result['training_producers']} == {'hold_b', 'hold_e'}
    with pytest.raises(ContractError, match='independent cost validation'):
        require_cost_validation(c, ['a'], groups, validation_producer_id='hold_a')


def test_nested_clean_producers_pass_before_any_training(tmp_path):
    p = three_folds(tmp_path)
    nested = artifact(tmp_path, p, 'nested', fit=('e',))
    outer = artifact(tmp_path, p, 'outer', fit=('b', 'e'))
    prep = artifact(tmp_path, p, 'prep', kind='preprocessor', fit=('b',))
    c = ExperimentContract(p, tmp_path, [nested, outer, prep])
    result = require_cost_validation(c, ['a'], [{'recordings': ['b'], 'predictor_id': 'nested'}],
                                     validation_producer_id='outer', preprocessing_ids=['prep'])
    assert result['eligible_under_declared_lineage']
    assert not result['independent_calibration_certificate']


def test_global_head_is_in_sample_even_when_each_producer_was_oof(tmp_path):
    p = three_folds(tmp_path)
    clean = artifact(tmp_path, p, 'clean', fit=('b', 'e'))
    head = artifact(tmp_path, p, 'head', kind='risk_head', fit=('a', 'b', 'e'), parents=('clean',))
    c = ExperimentContract(p, tmp_path, [clean, head])
    result = audit_prediction_lineage(c, 'head', ['a'])
    assert not result['eligible_under_declared_lineage']
    assert result['exposures'] == [{'artifact_id': 'head', 'kind': 'risk_head',
                                  'field': 'fit_recordings', 'recordings': ['a'],
                                  'physical_scenes': ['a'], 'is_root': True}]


def test_preprocessor_parent_exposure_is_not_hidden(tmp_path):
    p = three_folds(tmp_path)
    prep = artifact(tmp_path, p, 'prep', kind='preprocessor', fit=('a',))
    child = artifact(tmp_path, p, 'child', fit=('e',), parents=('prep',))
    outer = artifact(tmp_path, p, 'outer', fit=('b', 'e'))
    c = ExperimentContract(p, tmp_path, [prep, child, outer])
    r = audit_cost_validation(c, ['a'], [{'recordings': ['b'], 'predictor_id': 'child'}],
                             validation_producer_id='outer')
    assert not r['eligible_under_declared_lineage']
    assert r['training_producers'][0]['exposures'][0]['artifact_id'] == 'prep'
    assert not r['training_producers'][0]['exposures'][0]['is_root']


def test_refitting_head_does_not_repair_leaky_global_normalizer(tmp_path):
    p = three_folds(tmp_path)
    inner = artifact(tmp_path, p, 'inner', fit=('e',))
    outer = artifact(tmp_path, p, 'outer', fit=('b', 'e'))
    prep = artifact(tmp_path, p, 'prep', kind='preprocessor', fit=('a', 'b', 'e'))
    c = ExperimentContract(p, tmp_path, [inner, outer, prep])
    r = audit_cost_validation(c, ['a'], [{'recordings': ['b'], 'predictor_id': 'inner'}],
                             validation_producer_id='outer', preprocessing_ids=['prep'])
    assert not r['eligible_under_declared_lineage']
    assert r['preprocessors'][0]['exposures'][0]['recordings'] == ['a']


@pytest.mark.parametrize('fault', ['checkpoint', 'manifest'])
def test_identity_failure_is_not_downgraded_to_expected_exposure(tmp_path, fault):
    p = fixture_contract(tmp_path)
    a = artifact(tmp_path, p, 'leaky', fit=('a', 'b'))
    c = ExperimentContract(p, tmp_path, [a])
    if fault == 'checkpoint':
        (tmp_path / 'leaky.bin').write_bytes(b'changed')
    else:
        c.artifacts['leaky']['fit_recordings'] = ['b']
    with pytest.raises(ContractError, match='identity'):
        audit_prediction_lineage(c, 'leaky', ['a'])


def test_validation_prediction_must_also_be_held_out(tmp_path):
    p = three_folds(tmp_path)
    inner = artifact(tmp_path, p, 'inner', fit=('e',))
    full = artifact(tmp_path, p, 'full', fit=('a', 'b', 'e'))
    c = ExperimentContract(p, tmp_path, [inner, full])
    r = audit_cost_validation(c, ['a'], [{'recordings': ['b'], 'predictor_id': 'inner'}],
                             validation_producer_id='full')
    assert not r['eligible_under_declared_lineage']
    assert r['upstream_training_scene_overlap'] == []
    assert not r['validation_producer']['eligible_under_declared_lineage']


def test_direct_training_overlap_and_non_oof_groups_are_refused(tmp_path):
    p = fixture_contract(tmp_path)
    outer = artifact(tmp_path, p, 'outer', fit=('b',))
    c = ExperimentContract(p, tmp_path, [outer])
    r = audit_cost_validation(c, ['a'], [{'recordings': ['a'], 'predictor_id': 'outer'}],
                             validation_producer_id='outer')
    assert r['direct_training_scene_overlap'] == ['a']
    assert not r['eligible_under_declared_lineage']
    with pytest.raises(ContractError, match='exposure'):
        audit_cost_validation(c, ['a'], [{'recordings': ['b'], 'predictor_id': 'outer'}],
                              validation_producer_id='outer')


@pytest.mark.parametrize('targets', [[], ['d'], ['missing']])
def test_missing_or_wrong_role_validation_is_not_allowed(tmp_path, targets):
    p = fixture_contract(tmp_path)
    c = ExperimentContract(p, tmp_path, [artifact(tmp_path, p, 'model')])
    with pytest.raises(ContractError):
        audit_prediction_lineage(c, 'model', targets)


def test_held_fold_includes_its_other_recordings(tmp_path):
    p = three_folds(tmp_path)
    p['fit_folds']['e'] = 0
    approve(p)
    producer = artifact(tmp_path, p, 'model', fit=('e',))
    c = ExperimentContract(p, tmp_path, [producer])
    r = audit_prediction_lineage(c, 'model', ['a'])
    assert r['held_recordings'] == ['a', 'e']
    assert r['exposures'][0]['recordings'] == ['e']
    assert not r['eligible_under_declared_lineage']


def test_same_physical_scene_alias_cannot_hide_exposure(tmp_path):
    import numpy as np
    p = three_folds(tmp_path)
    write_recording(tmp_path / 'alias', np.array([[j*10, 4, j*.3, 7] for j in range(25)]),
                    {'id': 'alias', 'physical_scene': 'a'})
    p['records']['alias'] = dict(p['records']['a'], cache_path='alias',
                                 metadata_sha256=file_digest(tmp_path / 'alias/metadata.json'))
    p['assignments']['alias'] = 'fit'
    p['fit_folds']['alias'] = 0
    approve(p)
    producer = artifact(tmp_path, p, 'model', fit=('alias',))
    c = ExperimentContract(p, tmp_path, [producer])
    r = audit_prediction_lineage(c, 'model', ['a'])
    assert r['exposures'][0]['recordings'] == ['alias']
    assert r['exposures'][0]['physical_scenes'] == ['a']
    assert not r['eligible_under_declared_lineage']
