"""Dedicated-role empirical calibration, not a population-risk certificate."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_unification.m3w_model_selection_inputs import INPUT_FIELDS
from src.data_unification.m3w_selection_feature_bridge import FIELDS
from src.world_model import m3w_frame_action_allocation as allocation

MULTIPLIERS = tuple(2 ** k for k in range(13))


def require_calibration_role(roles, member):
    if roles.get('status') != 'restricted_source_training_admitted_reserved_roles_closed':
        raise ValueError('Original restricted role manifest required')
    rows = [r for r in roles['recordings'] if r['source_member'] == member]
    if len(rows) != 1:
        raise ValueError('Unique recording identity required')
    row = rows[0]
    if (row['role'] != 'risk_calibration_reserved' or row['training_access']
            or row['predictive_readout_access']):
        raise PermissionError('This new operator admits only the dedicated risk-calibration role')
    return row


def load_record(directory, roles, member, registration_sha256):
    row = require_calibration_role(roles, member)
    directory = Path(directory)
    receipt = json.loads((directory / 'receipt.json').read_text())
    expected = {'input_' + k for k in INPUT_FIELDS} | {'feature_' + k for k in FIELDS} | {'index_query_frames'}
    if (receipt.get('role') != 'risk_calibration_inputs_only'
            or receipt.get('source_member') != member or receipt.get('rows_sha256') != row['rows_sha256']
            or receipt.get('registration_sha256') != registration_sha256
            or set(receipt['arrays']) != expected):
        raise ValueError('Exact input-only calibration receipt required; labels are a separate interface')
    result = {}
    for field, ref in receipt['arrays'].items():
        if ref['name'] != field + '.npy':
            raise ValueError('External/label paths are forbidden')
        path = directory / ref['name']
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Input cache changed')
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(value.shape) != ref['shape'] or str(value.dtype) != ref['dtype']:
            raise ValueError('Input shape/dtype changed')
        result[field] = value
    return ({k: result['input_' + k] for k in INPUT_FIELDS},
            {k: result['feature_' + k] for k in FIELDS}, result['index_query_frames'])


def coefficient_view(prediction, allowed, multiplier):
    """Tighten both risk budgets without changing the native B-H objective."""
    p, _ = allocation.validate(prediction, allowed)
    if multiplier not in MULTIPLIERS:
        raise ValueError('Registered binary multiplier required')
    out = p.copy()
    out[..., 2] /= multiplier
    out[..., 3] /= multiplier
    # This is an optimization-coefficient view, not a coherent cost distribution.
    np.testing.assert_array_equal(out[..., 0] - out[..., 1], p[..., 0] - p[..., 1])
    return out


def select_multiplier(curves, required_views):
    if required_views <= 0 or [c['multiplier'] for c in curves] != list(MULTIPLIERS):
        raise ValueError('Complete preregistered grid and nonempty full cohort required')
    eligible = []
    for row in curves:
        passed = all(row['policies'][a]['required_record_views'] == required_views
                     and row['policies'][a]['supported_record_views'] == required_views
                     for a in ('independent', 'joint_matched'))
        if passed:
            eligible.append(row['multiplier'])
    return dict(selected_multiplier=min(eligible) if eligible else None,
                calibration_supported=bool(eligible),
                selection_rule='least_multiplier_passing_every_record_view_for_both_required_policies',
                global_floor_fallback_if_unsupported=not bool(eligible),
                empty_support_is_safe=False, empirical_calibration_not_independent_test=True,
                population_risk_certificate=False, confirmation_access=False,
                deployment_changed=False, stage5c_executed=False, smc_enabled=False)
