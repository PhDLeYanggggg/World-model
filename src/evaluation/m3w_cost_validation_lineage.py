"""Pre-fit lineage checks for cost-head holdouts; not a calibration certificate.

An OOF forecast is excluded from its own row's scene, but can still expose a
different scene proposed for cost-head validation. This module explains that
recursive exposure using the existing experiment contract, without model or
target-array access.
"""
from __future__ import annotations

from src.evaluation.m3w_experiment_contract import ContractError


def _held_recordings(contract, recordings):
    contract._assert_frozen()
    if (not isinstance(recordings, list) or not recordings or
            len(recordings) != len(set(recordings)) or
            any(contract.protocol['assignments'].get(r) != 'fit' for r in recordings)):
        raise ContractError('Cost validation requires unique nonempty fit-role recordings')
    folds = contract.protocol['fit_folds']
    held = {folds[r] for r in recordings}
    return sorted(r for r, fold in folds.items() if fold in held)


def audit_prediction_lineage(contract, artifact_id, recordings):
    """Return every declared exposure, and independently check the old guard."""
    held = _held_recordings(contract, recordings)
    target_scenes = contract.scene_set(held)
    closure = sorted(contract._closure(artifact_id))
    # Check all bytes before classifying a failure as an expected overlap.
    for name in closure:
        contract._verify_artifact(contract.artifacts[name])
    exposures = []
    for name in closure:
        a = contract.artifacts[name]
        for field in ('fit_recordings', 'selection_recordings', 'calibration_recordings'):
            overlap = sorted(r for r in a[field] if
                             contract.protocol['records'][r]['physical_scene'] in target_scenes)
            if overlap:
                exposures.append(dict(artifact_id=name, kind=a['kind'], field=field,
                    recordings=overlap, physical_scenes=sorted(contract.scene_set(overlap)),
                    is_root=name == artifact_id))
    try:
        contract.assert_prediction_use(artifact_id, recordings, purpose='oof_risk_training')
    except ContractError as exc:
        if not exposures or 'prediction target has upstream exposure' not in str(exc):
            raise
        accepted = False
    else:
        if exposures:
            raise ContractError('Recursive exposure diagnosis disagrees with prediction guard')
        accepted = True
    return dict(artifact_id=artifact_id, held_recordings=held,
                held_physical_scenes=sorted(target_scenes), ancestor_ids=closure,
                exposures=exposures, eligible_under_declared_lineage=accepted,
                existing_contract_guard_agrees=True)


def audit_cost_validation(contract, validation_recordings, training_groups, *,
                          validation_producer_id, preprocessing_ids=()):
    """Check a proposed head refit without allocating/fitting any new head.

    `training_groups` contains only recording IDs and the frozen producer ID.
    A new head and its newly fitted normalizer must use these training rows only.
    Already fitted preprocessors must be supplied explicitly for lineage checks.
    """
    held = _held_recordings(contract, validation_recordings)
    held_scenes = contract.scene_set(held)
    if not training_groups:
        raise ContractError('Cost validation needs nonempty OOF training groups')
    training_recordings, training_producers = set(), set()
    for group in training_groups:
        records = group['recordings']
        _held_recordings(contract, records)
        if training_recordings & set(records):
            raise ContractError('Repeated cost-training recording group')
        name = group['predictor_id']
        if contract.artifacts.get(name, {}).get('kind') != 'forecaster':
            raise ContractError('Cost-training producer must be a forecaster')
        contract.assert_prediction_use(name, records, purpose='oof_risk_training')
        training_recordings.update(records)
        training_producers.add(name)
    if contract.artifacts.get(validation_producer_id, {}).get('kind') != 'forecaster':
        raise ContractError('Validation producer must be a forecaster')
    if len(preprocessing_ids) != len(set(preprocessing_ids)):
        raise ContractError('Duplicate preprocessor identity')
    for name in preprocessing_ids:
        if contract.artifacts.get(name, {}).get('kind') != 'preprocessor':
            raise ContractError('Explicit preprocessor identity required')
    producers = [audit_prediction_lineage(contract, name, validation_recordings)
                 for name in sorted(training_producers)]
    preprocessors = [audit_prediction_lineage(contract, name, validation_recordings)
                     for name in sorted(preprocessing_ids)]
    validation = audit_prediction_lineage(contract, validation_producer_id, validation_recordings)
    direct = sorted(held_scenes & contract.scene_set(training_recordings))
    indirect = sorted({s for r in producers + preprocessors for e in r['exposures']
                       for s in e['physical_scenes']})
    return dict(held_recordings=held, held_physical_scenes=sorted(held_scenes),
                proposed_training_recordings=sorted(training_recordings),
                direct_training_scene_overlap=direct,
                upstream_training_scene_overlap=indirect,
                training_producers=producers, preprocessors=preprocessors,
                validation_producer=validation,
                eligible_under_declared_lineage=(not direct and not indirect and
                                                 validation['eligible_under_declared_lineage']),
                head_refit_executed=False, requires_train_only_head_and_normalizer_refit=True,
                independent_calibration_certificate=False)


def require_cost_validation(contract, validation_recordings, training_groups, **kwargs):
    """Fail closed before a proposed refit falsely claims independent validation."""
    result = audit_cost_validation(contract, validation_recordings, training_groups, **kwargs)
    if not result['eligible_under_declared_lineage']:
        raise ContractError('Proposed independent cost validation has direct or upstream exposure')
    return result
