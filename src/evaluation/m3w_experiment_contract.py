"""Split/lineage preflight for new M3W experiments, not an independence certificate.

Approval and historical-use declarations require a human/source audit. This module
checks consistency and file identity; it cannot prove that declarations are true.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_unification.m3w_causal_recordings import RecordingWindows, PROTOCOL_RAW, PROTOCOL_STEPS
from src.evaluation.m3w_recording_lineage import sha256 as file_digest


ROLES = {'fit', 'development', 'calibration', 'confirmation', 'excluded'}
ARTIFACT_KINDS = {'forecaster', 'risk_head', 'preprocessor', 'goal_prototypes', 'policy', 'calibrator'}


class ContractError(ValueError):
    pass


def protocol_digest(protocol: dict) -> str:
    content = {k: v for k, v in protocol.items() if k != 'approval'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ContractError(reason)


def _names(values, label):
    _require(isinstance(values, list) and all(isinstance(v, str) and v for v in values), label)
    _require(len(values) == len(set(values)), f'Duplicate {label}')
    return set(values)


class ExperimentContract:
    def __init__(self, protocol: dict, root: Path, artifacts: list[dict] | None = None):
        # Detach mutable caller dictionaries so approved values cannot drift in place.
        self.protocol = json.loads(json.dumps(protocol, allow_nan=False))
        self.root = Path(root).resolve()
        self.digest = protocol_digest(self.protocol)
        self._validate_protocol()
        self.artifacts = {}
        for record in artifacts or []:
            record = json.loads(json.dumps(record, allow_nan=False))
            name = record['id']
            _require(name not in self.artifacts, 'Duplicate artifact identity')
            self.artifacts[name] = record
        self._validate_artifacts()
        self._artifact_manifest_digest = self._artifact_digest()

    @property
    def confirmation_eligible_by_declaration(self):
        # Eligibility under declarations, not empirical proof of IID or untouched data.
        return self.protocol['scope'] == 'confirmatory'

    def _validate_protocol(self):
        p = self.protocol
        approval = p.get('approval') or {}
        _require(p.get('schema_version') == 1, 'Unsupported protocol schema')
        _require(p.get('status') == 'approved' and approval.get('approved_by') and
                 approval.get('decision_reference'), 'Explicit protocol approval required')
        _require(approval.get('protocol_sha256') == self.digest, 'Approval digest changed')
        for relative, digest in p.get('bindings', {}).items():
            path = self._path(relative)
            _require(path.is_file() and file_digest(path) == digest, f'Changed protocol binding: {relative}')
        _require(p.get('scope') in {'exploratory', 'confirmatory'}, 'Explicit evidence scope required')
        for stage in ('calibration', 'confirmation'):
            _require(p.get(f'{stage}_receipt'), f'Explicit protocol-bound {stage} receipt required')
            self._path(p[f'{stage}_receipt'])
        _require(self._path(p['calibration_receipt']) != self._path(p['confirmation_receipt']),
                 'Separate calibration and confirmation receipts required')
        records, roles = p['records'], p['assignments']
        _require(records and set(records) == set(roles), 'Complete recording role assignments required')
        _require(set(roles.values()) <= ROLES, 'Unknown recording role')
        _require(ROLES - {'excluded'} <= set(roles.values()), 'All four data roles must be nonempty')
        scenes = {}
        for name, record in records.items():
            _require(isinstance(record.get('physical_scene'), str) and record['physical_scene'],
                     'Reviewed physical scene identity required')
            _require(record.get('review_evidence'), 'Historical-use review evidence required')
            _require(record.get('historical_use') in {'unknown', 'development_exposed', 'unexposed_reviewed'},
                     'Unknown historical-use status')
            _require(record.get('cache_path') and record.get('metadata_sha256'), 'Cache binding required')
            if roles[name] != 'excluded':
                scenes.setdefault(record['physical_scene'], set()).add(roles[name])
            if p['scope'] == 'confirmatory' and roles[name] in {'calibration', 'confirmation'}:
                _require(record['historical_use'] == 'unexposed_reviewed',
                         f'{name}: historical exposure incompatible with independent {roles[name]}')
        _require(all(len(v) == 1 for v in scenes.values()), 'A physical scene crosses data roles')
        fit = {r for r, role in roles.items() if role == 'fit'}
        folds = p['fit_folds']
        _require(set(folds) == fit and len(set(folds.values())) >= 2 and
                 all(type(f) is int and f >= 0 for f in folds.values()), 'Complete multi-group crossfit folds required')
        for scene in scenes:
            assigned = {folds[r] for r in fit if records[r]['physical_scene'] == scene}
            _require(len(assigned) <= 1, 'Physical scene crosses crossfit folds')
        task = p['task']
        _require(type(task.get('history_steps')) is int and task['history_steps'] >= 3,
                 'Explicit past history length required')
        _require(task.get('prediction_unit') in {'observation_steps', 'raw_annotation_frames'} and
                 type(task.get('horizon')) is int and task['horizon'] > 0,
                 'Explicit supported time unit and horizon required; seconds are not verified')
        _require(task.get('coordinate_claim') == 'dataset_local_unverified', 'Unverified metric claim')
        _require(task.get('primary_metric') in {'ade', 'fde'} and
                 task.get('aggregation') in {'equal_physical_scene', 'equal_recording', 'agent_window'},
                 'Explicit metric and aggregation required')
        risk = p['risk']
        _require(0 < risk['delta'] < 1 and risk.get('risks') and risk.get('easy_definition'),
                 'Explicit risks and easy definition required')
        _require(np.isfinite(risk['easy_degradation_max']) and risk['easy_degradation_max'] >= 0,
                 'Explicit easy risk tolerance required')
        for spec in risk['risks']:
            lo, hi, tol = (spec[k] for k in ('lower', 'upper', 'tolerance'))
            _require(spec.get('name') and np.isfinite([lo, hi, tol]).all() and lo < hi and lo <= tol <= hi,
                     'Finite bounded risk specification required')
        seeds = p['seeds']
        _require(len(seeds) >= 3 and len(set(seeds)) == len(seeds) and
                 all(type(s) is int for s in seeds), 'At least three explicit distinct formal seeds required')
        _require(p.get('bootstrap_unit') == 'physical_scene' and
                 type(p.get('bootstrap_resamples')) is int and p['bootstrap_resamples'] >= 2000,
                 'Explicit physical-scene bootstrap with at least 2000 resamples required')
        byte_groups = {}
        for name, record in records.items():
            directory = self._path(record['cache_path'])
            metadata = directory / 'metadata.json'
            _require(metadata.is_file() and file_digest(metadata) == record['metadata_sha256'],
                     'Changed cache metadata')
            content = json.loads(metadata.read_text())
            _require(content['id'] == name and content['physical_scene'] == record['physical_scene'],
                     'Reviewed physical scene disagrees with cache identity')
            if roles[name] == 'excluded':
                continue
            digests = [content['artifacts']['points.npy']['sha256']]
            digests += [entry['sha256'] for entry in content.get('files', [])]
            for digest in digests:
                byte_groups.setdefault(digest, set()).add(roles[name])
        _require(all(len(v) == 1 for v in byte_groups.values()), 'Byte-identical data cross roles despite renamed scenes')

    def scene_set(self, recordings):
        _require(set(recordings) <= set(self.protocol['records']), 'Unknown recording identity')
        return {self.protocol['records'][r]['physical_scene'] for r in recordings}

    def _path(self, relative):
        path = (self.root / relative).resolve()
        _require(path.is_relative_to(self.root), 'Artifact path escapes experiment workspace')
        return path

    def _validate_artifacts(self):
        seen_hashes = {}
        for name, record in self.artifacts.items():
            _require(record.get('kind') in ARTIFACT_KINDS, 'Unknown artifact kind')
            _require(record.get('lineage_complete') is True, 'Incomplete artifact lineage')
            _require(record.get('protocol_sha256') == self.digest, 'Artifact belongs to another protocol')
            for field, role in [('fit_recordings', 'fit'), ('selection_recordings', 'development'),
                                ('calibration_recordings', 'calibration')]:
                used = _names(record[field], field)
                _require(used <= set(self.protocol['records']) and
                         all(self.protocol['assignments'][r] == role for r in used),
                         f'{name}: illegal {field} exposure')
            _require(not record['calibration_recordings'] or record['kind'] == 'calibrator',
                     'Only a calibrator may declare calibration use; not a predictor refit')
            parents = _names(record['parents'], 'parents')
            _require(parents <= self.artifacts.keys(), 'Missing artifact parent')
            identity = tuple(tuple(sorted(record[k])) for k in
                             ('fit_recordings', 'selection_recordings', 'calibration_recordings', 'parents'))
            if record['sha256'] in seen_hashes:
                _require(seen_hashes[record['sha256']] == identity, 'Same artifact bytes have conflicting lineage')
            seen_hashes[record['sha256']] = identity
            self._verify_artifact(record)
        for name in self.artifacts:
            self._closure(name)

    def _verify_artifact(self, record):
        path = self._path(record['path'])
        _require(path.is_file() and file_digest(path) == record['sha256'], 'Changed artifact identity')

    def _artifact_digest(self):
        return hashlib.sha256(json.dumps(self.artifacts, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _assert_frozen(self):
        _require(protocol_digest(self.protocol) == self.digest, 'Approved protocol identity changed in memory')
        _require(self._artifact_digest() == self._artifact_manifest_digest, 'Artifact lineage identity changed in memory')
        for relative, digest in self.protocol.get('bindings', {}).items():
            path = self._path(relative)
            _require(path.is_file() and file_digest(path) == digest, f'Changed protocol binding: {relative}')

    def _closure(self, name, stack=()):
        _require(name in self.artifacts, 'Unknown artifact')
        _require(name not in stack, 'Artifact dependency cycle')
        result = {name}
        for parent in self.artifacts[name]['parents']:
            result |= self._closure(parent, (*stack, name))
        return result

    def assert_prediction_use(self, artifact_id, recordings, *, purpose):
        self._assert_frozen()
        role = {'oof_risk_training': 'fit', 'development': 'development',
                'calibration': 'calibration', 'confirmation': 'confirmation'}.get(purpose)
        _require(role and recordings and all(self.protocol['assignments'].get(r) == role for r in recordings),
                 'Prediction purpose does not match recording role')
        target_scenes = self.scene_set(recordings)
        if purpose == 'oof_risk_training':
            folds = self.protocol['fit_folds']
            held_out = {folds[r] for r in recordings}
            target_scenes = self.scene_set([r for r, fold in folds.items() if fold in held_out])
        for name in self._closure(artifact_id):
            artifact = self.artifacts[name]
            self._verify_artifact(artifact)
            used = artifact['fit_recordings']
            if purpose != 'development':
                used = used + artifact['selection_recordings'] + artifact['calibration_recordings']
            _require(not target_scenes & self.scene_set(used), f'{name}: prediction target has upstream exposure')

    def open_recording(self, name, *, purpose, claim_path=None):
        self._assert_frozen()
        _require(purpose in ROLES - {'excluded'} and self.protocol['assignments'].get(name) == purpose,
                 'Reader purpose violates data role')
        if purpose in {'calibration', 'confirmation'}:
            _require(claim_path is not None, f'Frozen {purpose} claim required before label access')
            _check_claim_path(claim_path, self, purpose)
            receipt = json.loads(Path(claim_path).read_text())
            _check_claim(receipt, self, receipt['artifact_ids'], purpose)
        record = self.protocol['records'][name]
        directory = self._path(record['cache_path'])
        _require(file_digest(directory / 'metadata.json') == record['metadata_sha256'], 'Changed cache metadata')
        reader = RecordingWindows(directory)
        _require(reader.metadata['id'] == name and reader.metadata['physical_scene'] == record['physical_scene'],
                 'Recording cache identity disagrees with reviewed catalog')
        schema = reader.metadata['schema']
        _require(all(schema.get(k) is False for k in ('future_inputs', 'central_velocity_used', 'legacy_teacher_inputs')),
                 'Causal input schema not verified')
        task = self.protocol['task']
        _require(reader.metadata['history_steps'] == task['history_steps'], 'History schema mismatch')
        code, field = ((PROTOCOL_STEPS, 'future_steps') if task['prediction_unit'] == 'observation_steps'
                       else (PROTOCOL_RAW, 'horizon_raw'))
        ids = np.flatnonzero((reader.index['protocol'] == code) & (reader.index[field] == task['horizon']))
        _require(len(ids) > 0, 'No exact eligible windows for the approved time protocol')
        return reader, ids


def _claim_identity(contract, artifact_ids, stage):
    contract._assert_frozen()
    _require(artifact_ids and len(set(artifact_ids)) == len(artifact_ids), 'Fixed nonempty evaluation family required')
    targets = [r for r, role in contract.protocol['assignments'].items() if role == stage]
    closure = set()
    for name in artifact_ids:
        contract.assert_prediction_use(name, targets, purpose=stage)
        closure |= contract._closure(name)
    return {'protocol_sha256': contract.digest, 'artifact_ids': sorted(artifact_ids), 'stage': stage,
            'artifact_hashes': {n: contract.artifacts[n]['sha256'] for n in sorted(closure)},
            'evaluation_recordings': sorted(targets), 'scope': contract.protocol['scope']}


def _check_claim(receipt, contract, artifact_ids, stage):
    expected = _claim_identity(contract, artifact_ids, stage)
    _require(all(receipt.get(k) == v for k, v in expected.items()), f'Frozen {stage} identity changed')
    _require(receipt.get('state') == 'started', f'{stage} already completed or invalid state')


def _check_claim_path(path, contract, stage):
    _require(Path(path).resolve() == contract._path(contract.protocol[f'{stage}_receipt']),
             f'{stage} receipt path differs from approved protocol')


def _claim_round(path, contract, artifact_ids, *, stage, resume):
    path = Path(path)
    _check_claim_path(path, contract, stage)
    identity = _claim_identity(contract, artifact_ids, stage)
    if resume:
        _require(path.is_file(), f'Cannot resume a missing {stage} claim')
        receipt = json.loads(path.read_text())
        _check_claim(receipt, contract, artifact_ids, stage)
        return receipt
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {**identity, 'state': 'started', 'fresh_independent_evidence': False,
               'note': 'Reservation only; no predictions or result verified'}
    try:
        # Exclusive creation prevents concurrent first claims at this receipt path.
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    except FileExistsError as exc:
        raise ContractError(f'{stage} already claimed; resume identical work only') from exc
    return receipt


def _finish_round(path, contract, artifact_ids, result_path, *, stage):
    path = Path(path)
    _check_claim_path(path, contract, stage)
    receipt = json.loads(path.read_text())
    _check_claim(receipt, contract, artifact_ids, stage)
    _require(Path(result_path).is_file(), 'Cannot complete without a result artifact')
    receipt.update(state='completed', result_path=str(result_path), result_sha256=file_digest(result_path),
                   note='Execution receipt only; scientific result correctness still requires audit')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    temporary.replace(path)
    return receipt


def claim_confirmation(path: Path, contract: ExperimentContract, artifact_ids: list[str], *, resume=False):
    return _claim_round(path, contract, artifact_ids, stage='confirmation', resume=resume)


def finish_confirmation(path: Path, contract: ExperimentContract, artifact_ids: list[str], result_path: Path):
    return _finish_round(path, contract, artifact_ids, result_path, stage='confirmation')


def claim_calibration(path: Path, contract: ExperimentContract, artifact_ids: list[str], *, resume=False):
    return _claim_round(path, contract, artifact_ids, stage='calibration', resume=resume)


def finish_calibration(path: Path, contract: ExperimentContract, artifact_ids: list[str], result_path: Path):
    return _finish_round(path, contract, artifact_ids, result_path, stage='calibration')
