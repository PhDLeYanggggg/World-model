"""Label-aware recording capacity under whole-query forecast options."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.evaluation.m3w_selection_forecast_readout import errors, policy


def prepare(cv, ols, y, valid, offsets, slots, paths, easy_cut):
    if (slots.ndim != 3 or paths.shape != (*slots.shape, *cv.shape[1:])
            or len(offsets) != slots.shape[1]+1 or offsets[0] != 0 or offsets[-1] != len(cv)
            or np.any(np.diff(offsets) <= 0) or np.any(slots < -1)):
        raise ValueError('Misaligned whole-query options')
    reference, _ = errors(cv, y, valid)
    easy = valid.any(1) & (reference > 0.) & (reference <= easy_cut)
    weights = np.repeat(1./np.diff(offsets), np.diff(offsets))
    ids = slots >= 0
    query = np.broadcast_to(np.arange(slots.shape[1])[None, :, None], slots.shape)[ids]
    row = slots[ids]
    if np.any(row < offsets[query]) or np.any(row >= offsets[query+1]):
        raise ValueError('Option refers to another query')
    if slots.shape[-1] == 2 and np.any(ids.all(-1) & (slots[..., 0] == slots[..., 1])):
        raise ValueError('Repeated agent in an option')
    m = moments(cv[row], paths[ids], y[row], valid[row], easy_cut)
    if (m['radius'] <= 0).any():
        raise ValueError('A switch slot must change a forecast')
    unknown = ~m['known']
    ols_delta = np.linalg.norm(cv[row]-ols[row], axis=-1).max(1) - np.linalg.norm(paths[ids]-ols[row], axis=-1).max(1)
    values = np.column_stack((m['all_excess'], m['easy_excess'], m['easy_signed'],
                              m['ref_easy'] > 0, weights[row]*m['utility'],
                              weights[row]*np.where(unknown, ols_delta, m['diff'])))
    options = np.broadcast_to(np.arange(slots.shape[0]*slots.shape[1]).reshape(*slots.shape[:2], 1),
                              slots.shape)[ids]
    terms = np.zeros((slots.shape[0]*slots.shape[1], values.shape[1]))
    np.add.at(terms, options, values)
    keep = ids.any(-1).ravel()
    base = comparison_masses(ols, cv, cv, y, valid, weights)
    return dict(terms=terms[keep], query=np.tile(np.arange(slots.shape[1]), slots.shape[0])[keep],
        arm=np.repeat(np.arange(slots.shape[0]), slots.shape[1])[keep],
        cap=BUDGET*reference[easy].sum(), easy_rows=int(easy.sum()), weights=weights,
        offset_ols=base['known_delta']-base['unknown_radius'], reference_mass=base['reference'])


def solve(prepared, objective, allowed_arms, seconds=1.):
    use = np.flatnonzero(np.isin(prepared['arm'], allowed_arms))
    m, q = prepared['terms'][use], prepared['query'][use]
    gain_col = {'cv': 4, 'ols4': 5}[objective]
    offset = 0. if objective == 'cv' else prepared['offset_ols']
    if not len(m) or not (m[:, 3] > 0.).any():
        return None, dict(status='no_easy_option_support', proved_infeasible=True, optimum_proved=False)
    unit = np.abs(m[:, :3]).max(0).clip(1.)
    n = len(m)
    groups, group = np.unique(q, return_inverse=True)
    membership = coo_array((np.ones(n), (group, np.arange(n))), shape=(len(groups), n)).tocsc()
    matrix = vstack((membership, csc_array((m[:, :3]/unit).T), csc_array(m[None, :, 3])), format='csc')
    low = np.r_[np.full(len(groups)+3, -np.inf), 1.]
    high = np.r_[np.ones(len(groups)), 0., 0., prepared['cap']/unit[2], np.inf]
    obj = m[:, gain_col]
    scale = max(1., np.abs(obj).max())
    sol = milp(-obj/scale, integrality=np.ones(n), bounds=Bounds(0., 1.),
        constraints=LinearConstraint(matrix, low, high),
        options=dict(time_limit=seconds, mip_rel_gap=0.))
    dual = getattr(sol, 'mip_dual_bound', None)
    info = dict(status={0:'optimal', 1:'timed', 2:'infeasible'}.get(sol.status, 'unresolved'),
        optimum_proved=bool(sol.status == 0), proved_infeasible=bool(sol.status == 2),
        upper_gain_mass=float(offset-dual*scale) if dual is not None and np.isfinite(dual) else None)
    if sol.x is None:
        return None, info
    take = sol.x > .5
    if len(np.unique(q[take])) != int(take.sum()) or m[take, 3].sum() < 1:
        info.update(status='unresolved_integral_postcheck', optimum_proved=False)
        return None, info
    info['witness_gain_mass'] = float(offset+obj[take].sum())
    return use[take], info


def audit(cv, ols, y, valid, offsets, slots, paths, recording, easy_cut, hard_cut):
    model = prepare(cv, ols, y, valid, offsets, slots, paths, easy_cut)
    outputs = {}
    for label, arms in (('joint_only', (3,)), ('four_option', (0, 1, 2, 3))):
        for name, ref in (('cv', cv), ('ols4', ols)):
            take, info = solve(model, name, arms)
            info['verified_witness'] = False
            if take is not None:
                pred = cv.copy()
                for i in take:
                    arm, q = model['arm'][i], model['query'][i]
                    use = slots[arm, q] >= 0
                    pred[slots[arm, q, use]] = paths[arm, q, use]
                result = policy(pred, cv, cv, y, valid, np.full(len(cv), recording),
                                easy_cut=easy_cut, hard_cut=hard_cut)['recordings'][recording]
                masses = comparison_masses(ref, pred, cv, y, valid, model['weights'])
                lower = masses['known_delta']-masses['unknown_radius']
                np.testing.assert_allclose(lower, info['witness_gain_mass'], atol=1e-7, rtol=1e-10)
                if result['supported']:
                    info.update(verified_witness=True, witness_switches=result['selected'],
                                witness_positive=bool(lower > 0.), witness_gain_mass=float(lower))
                else:
                    info.update(status='unresolved_original_risk_postcheck', optimum_proved=False)
            outputs[f'{label}_vs_{name}'] = info
    return dict(easy_rows=model['easy_rows'], reference_mass=model['reference_mass'], outcomes=outputs)
