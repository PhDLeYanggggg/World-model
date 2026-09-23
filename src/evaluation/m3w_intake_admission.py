"""Read-only admission checks for diagnostic sources awaiting separate review.

Hash-bound declarations are checked, not authenticated. A scientific protocol
approval alone cannot settle source-use conditions or erase annotation findings.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_source_reservations import check_source_reservation


ACTIVE_ROLES = {'fit', 'development', 'calibration', 'confirmation'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_file(root, reference, bindings):
    require(isinstance(reference, dict) and isinstance(reference.get('path'), str)
            and bool(reference['path']) and isinstance(reference.get('sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', reference['sha256']) is not None,
            'Invalid intake evidence binding')
    path = (root / reference['path']).resolve()
    require(path.is_relative_to(root), 'Intake evidence escapes workspace')
    require(path.is_file() and sha256(path) == reference['sha256'], 'Changed intake evidence')
    relative = str(path.relative_to(root))
    require(relative not in bindings or bindings[relative] == reference['sha256'], 'Conflicting intake evidence identity')
    bindings[relative] = reference['sha256']
    return path


def load_bound(root, reference, bindings):
    content = json.loads(bound_file(root, reference, bindings).read_text())
    require(isinstance(content, dict), 'Intake evidence must be a JSON object')
    return content


def validate_admission(root, name, record, metadata, role):
    """Refuse pending sources before opening recording arrays or future labels."""
    if role == 'excluded':
        return {}
    reservation_bindings = check_source_reservation(root, name, record, metadata, role)
    reference = record.get('intake_screen')
    if not reservation_bindings and 'source_conditions_review' not in metadata and reference is None:
        # Preserve the existing canonical protocol's declaration-based contract.
        # This does not retroactively approve every legacy source's conditions.
        return {}
    require(role in ACTIVE_ROLES, 'Unknown intake data role')
    require(reference is not None, f'{name}: source intake screen required')
    root, bindings = Path(root).resolve(), dict(reservation_bindings)
    screen = load_bound(root, reference, bindings)
    require(screen.get('schema_version') == 1 and screen.get('kind') == 'm3w_diagnostic_intake_screen',
            'Unsupported intake screen schema')
    require(isinstance(screen.get('recordings'), dict), 'Intake recording table must be a JSON object')
    entry = screen['recordings'].get(name)
    require(isinstance(entry, dict), 'Recording absent from intake screen')
    require(entry.get('metadata_sha256') == record['metadata_sha256']
            and entry.get('physical_scene') == record['physical_scene']
            and entry.get('dataset') == metadata.get('dataset')
            and (root / entry.get('cache_path', '')).resolve() == (root / record['cache_path']).resolve(),
            'Intake screen disagrees with recording identity')
    source_hashes = {f['path']: f['sha256'] for f in metadata.get('files', [])}
    require(bool(source_hashes) and entry.get('source_file_hashes') == source_hashes,
            'Intake source lineage differs from cache')
    audit = load_bound(root, screen.get('quality_audit'), bindings)
    source = load_bound(root, screen.get('source_manifest'), bindings)
    conversion = load_bound(root, screen.get('conversion_report'), bindings)
    require(audit.get('source_manifest_sha256') == screen['source_manifest']['sha256']
            and audit.get('build_report_sha256') == screen['conversion_report']['sha256'],
            'Intake audit has different upstream evidence')
    require(source.get('status') == 'verified' and source.get('upstream_commit')
            and source['upstream_commit'] == conversion.get('upstream_commit'),
            'Intake source version is not verified')
    manifest_hashes = {f['path']: f['sha256'] for f in source.get('files', [])}
    require(all(manifest_hashes.get(p) == h for p,h in source_hashes.items()), 'Unverified raw source identity')
    clips = [r for r in audit.get('clips', []) if r.get('recording') == name]
    built = [r for r in conversion.get('recordings', []) if r.get('id') == name]
    require(len(clips) == len(built) == 1 and clips[0].get('physical_scene') == record['physical_scene']
            and clips[0].get('points') == built[0].get('points') == metadata.get('points'),
            'Intake audit does not cover this recording')
    quarantined, flags = audit.get('quarantine_recordings'), audit.get('annotation_flags')
    require(isinstance(quarantined, list) and all(isinstance(v,str) for v in quarantined)
            and isinstance(flags, list) and all(isinstance(v,dict) for v in flags),
            'Explicit intake annotation dispositions required')
    relevant = [f for f in flags if f.get('recording') == name]
    require(name not in quarantined and not relevant, f'{name}: intake quality quarantine unresolved')
    require(entry.get('quality_disposition') == 'no_flags_in_limited_screen'
            and entry.get('blocking_findings') == [], 'Intake screen is not quality-admissible')
    decision = record.get('source_use_decision') or {}
    require(isinstance(decision, dict) and decision.get('status') == 'reviewed_for_declared_roles'
            and isinstance(decision.get('reviewed_by'), str) and bool(decision['reviewed_by'].strip())
            and isinstance(decision.get('decision_reference'), str) and bool(decision['decision_reference'].strip()),
            f'{name}: separate source-use review required')
    roles = decision.get('allowed_roles')
    require(isinstance(roles,list) and all(isinstance(v,str) for v in roles)
            and set(roles) <= ACTIVE_ROLES and len(roles) == len(set(roles)) and role in roles,
            'Source-use decision does not cover the assigned role')
    evidence = decision.get('evidence')
    require(isinstance(evidence,list) and bool(evidence), 'Hash-bound source-use decision evidence required')
    for reference in evidence:
        bound_file(root, reference, bindings)
    return bindings
