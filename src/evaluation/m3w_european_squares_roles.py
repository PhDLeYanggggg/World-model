"""Outcome-blind role assignment and fail-closed source access."""
import hashlib

ROLE_ORDER = ('source_training', 'risk_calibration_reserved', 'source_training',
              'risk_calibration_reserved', 'model_selection_reserved', 'confirmation_reserved')
SALT = 'm3w-european-squares-role-v1-2026-09-24'


def assign_roles(group_support):
    if len(group_support) != 36 or any(v < 0 for v in group_support.values()):
        raise ValueError('Expected 36 audited groups with nonnegative past support')
    ordered = sorted(group_support, key=lambda k: (-group_support[k], k))
    result = {}
    for start in range(0, len(ordered), 6):
        block = sorted(ordered[start:start+6], key=lambda k: hashlib.sha256((SALT+'|'+k).encode()).hexdigest())
        for key, role in zip(block, ROLE_ORDER):
            result[key] = dict(role=role, support_block=start//6,
                               assignment_digest=hashlib.sha256((SALT+'|'+key).encode()).hexdigest())
    return result


def require_source_training(manifest, source_member):
    if manifest.get('status') != 'restricted_source_training_admitted_reserved_roles_closed':
        raise ValueError('Source admission missing')
    matches = [r for r in manifest['recordings'] if r['source_member'] == source_member]
    if len(matches) != 1:
        raise ValueError('Unmapped or ambiguous recording')
    row = matches[0]
    if row['role'] != 'source_training' or not row['training_access']:
        raise PermissionError('Reserved role cannot enter source training')
    return row
