"""Monotone source-selected-set recalibration; no transfer coverage guarantee."""
import hashlib

import numpy as np

from src.world_model import m3w_component_calibration as parent
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds


def action_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def infer(p, env, moving, support, fitted, mode):
    """No outcomes or per-row future-availability masks at inference."""
    q = parent.adjust(p, env, fitted, mode)
    return q, parent.eligible(q, moving, support)


def fit(p, y, env, moving, support, recordings, mode, *, rounds=8, quantile=.9):
    if mode not in parent.MODES or not isinstance(rounds, int) or rounds < 1:
        raise ValueError('Fixed component arm and positive round cap required')
    rec = np.asarray(recordings).astype(str)
    raw = parent.eligible(p, moving, support)
    cal = parent.fit_margin(parent.record_scores(p, y, env, raw, rec), quantile)
    trace = []
    status = 'iteration_cap'
    for step in range(rounds):
        _, action = infer(p, env, moving, support, cal, mode)
        if not cal['supported'] or not action.any():
            status = 'unsupported' if not cal['supported'] else 'empty_selected_set'
            cal = dict(cal, supported=False)
            break
        # Refit TOTAL residual against the original predictions, not the
        # already-adjusted scores; adding cumulative residual twice is wrong.
        scores = parent.record_scores(p, y, env, action, rec)
        next_cal = parent.fit_margin(scores, quantile)
        next_cal['margins'] = np.maximum(cal['margins'], next_cal['margins']).tolist()
        _, next_action = infer(p, env, moving, support, next_cal, mode)
        if (next_action & ~action).any():
            raise AssertionError('Monotone margins must not introduce new actions')
        trace.append(dict(round=step+1, selected=int(action.sum()),
            next_selected=int(next_action.sum()), action_hash=action_hash(action),
            next_action_hash=action_hash(next_action), margins=cal['margins'],
            next_margins=next_cal['margins'],
            component_recording_counts=next_cal['component_recording_counts']))
        cal = next_cal
        if not cal['supported'] or not next_action.any():
            status = 'unsupported' if not cal['supported'] else 'empty_selected_set'
            cal = dict(cal, supported=False)
            break
        if np.array_equal(action, next_action):
            # The next residual fit sees the identical set and therefore
            # cannot change these componentwise-max total margins again.
            status = 'selected_set_fixed_point'
            break
    if status == 'iteration_cap':
        cal = dict(cal, supported=False)
    return dict(cal, status=status, trace=trace, round_cap=rounds,
        no_transfer_guarantee=True, source_selected_set_only=True)


def evaluate(p, y, env, moving, support, recordings, *, rounds=8, quantile=.9):
    """Whole-recording OOF: a held outcome cannot fit its own calibrator."""
    p, y, env, moving, support, rec = map(np.asarray,
        (p, y, env, moving, support, recordings))
    rec = rec.astype(str)
    old, old_oof = parent.calibrate(p, y, env, moving, support, rec, quantile)
    raw = parent.eligible(p, moving, support)
    rows = []
    for mode in parent.MODES:
        final = fit(p, y, env, moving, support, rec, mode, rounds=rounds, quantile=quantile)
        oof = np.zeros(len(p), bool)
        folds = []
        for name in np.unique(rec):
            held = rec == name
            train = ~held
            cal = fit(p[train], y[train], env[train], moving[train], support[train],
                rec[train], mode, rounds=rounds, quantile=quantile)
            assert name not in cal['recordings']
            _, oof[held] = infer(p[held], env[held], moving[held], support[held], cal, mode)
            folds.append(dict(held_recording=str(name), calibration=cal))
        _, resub = infer(p, env, moving, support, final, mode)
        _, old_resub = infer(p, env, moving, support, old['final'], mode)
        for role, a, b in (('source_oof', oof, old_oof[mode]),
                            ('source_resubstitution', resub, old_resub)):
            assert not (a & ~b).any() and not (a & ~raw).any()
            rows.append(dict(mode=mode, role=role,
                parent=completion_bounds(y, b, env), selected_set=completion_bounds(y, a, env),
                parent_action_hash=action_hash(b), action_hash=action_hash(a),
                removed=int((b & ~a).sum()), parent_selected=int(b.sum()),
                selected=int(a.sum())))
        rows[-2]['folds'] = folds
        rows[-1]['final'] = final
    return dict(rows=rows, parent_oof_action_hashes={m:action_hash(a) for m,a in old_oof.items()},
        parent_source=old['source'], parent_final=old['final'], independent_confirmation=False,
        new_neural_updates=0, transfer_evaluated=False)
