"""Calibration-only access, fixed activation ladder, and original risk checks."""
import numpy as np

from src.world_model import m3w_native_transport_policy as transport
from src.world_model import m3w_record_joint_actions as original


MARGINS = (0., .01, .03, .1)
GATES = tuple(f'gate_{j}' for j in range(len(MARGINS)))
ARMS = (*transport.MATCHED, *GATES)


def require_calibration(manifest, sequence):
    if (manifest['recording_role_index'].get(sequence) != 'calibration'
            or sequence not in manifest['assignments']['calibration']['recordings']
            or sequence in manifest['assignments']['confirmation']['recordings']):
        raise PermissionError('Calibration runner cannot access this recording')


def decide(history, bank, cv, ols, coef):
    x, scale = original.descriptors(history, bank, cv, np.full(len(cv), len(cv)))
    gain, all_excess, easy_excess, legal = original.decision_costs(x, coef, bank, cv, scale)
    matched = original.choose_query(history, cv, bank, ols, scale, gain, legal)[:4]
    joint = matched[-1]
    p = np.einsum('nad,adc->nac', x, coef)
    ref = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*scale[:, None]
    first = transport.gate(joint, gain, all_excess, easy_excess, ref)
    use = np.flatnonzero(first >= 0)
    v = (gain[use, first[use]]/scale[use]).sum()
    gates = [first if len(use) and v > m else np.full_like(joint, -1) for m in MARGINS]
    if len({int((a >= 0).sum()) for a in matched}) != 1:
        raise ValueError('Matched causal comparators changed the query budget')
    return np.stack((*matched, *gates))


def winner(summary):
    eligible = []
    for arm in GATES:
        r = summary['risk'][arm]
        effect = summary['effects'][f'{arm}_vs_cv']['completion_lower_pp']
        ci = effect['interval95']
        if (r['supported_views'] == r['required_views'] and r['switches'] > 0
                and ci is not None and ci[0] > 0.):
            eligible.append((-effect['point'], r['switches'], arm))
    return min(eligible)[2] if eligible else None
