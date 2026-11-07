"""Hold recipients fixed while enlarging the past-only trajectory slate."""
import numpy as np

from src.evaluation import m3w_joint_slate_capacity as base


SLATES = dict(frozen=tuple(range(4)), plus_ols4=tuple(range(8)),
              plus_common=(*range(4), *range(8, 12)), both=tuple(range(12)))


def augment(cv, ols, kinematic, offsets, slots, paths):
    if slots.shape[0] != 4 or not (cv.shape == ols.shape == kinematic.shape):
        raise ValueError('Expected four original arms and aligned forecasts')
    # Shift every OLS4 trajectory by the same query displacement before
    # restricting the replacement to the original intervention recipients.
    centers = np.add.reduceat((cv-ols)+(kinematic-ols), offsets[:-1], axis=0)
    common = ols + np.repeat(centers/(3.*np.diff(offsets)[:, None, None]),
                             np.diff(offsets), axis=0)
    used = slots >= 0
    extra_slots, extra_paths = [slots], [paths]
    for bank in (ols, common):
        alt = np.zeros_like(paths)
        alt[used] = bank[slots[used]]
        changed = np.zeros_like(used)
        changed[used] = (alt[used] != cv[slots[used]]).any(axis=(-1, -2))
        # A no-op cannot substitute for a matched-count option or buy support.
        matched = used & np.all(changed == used, axis=-1, keepdims=True)
        extra_slots.append(np.where(matched, slots, -1))
        extra_paths.append(np.where(matched[..., None, None], alt, 0.))
    return np.concatenate(extra_slots), np.concatenate(extra_paths)


def audit(cv, ols, kinematic, y, valid, offsets, slots, paths, recording, easy_cut, hard_cut):
    slots, paths = augment(cv, ols, kinematic, offsets, slots, paths)
    model = base.prepare(cv, ols, y, valid, offsets, slots, paths, easy_cut)
    outcomes = {}
    for slate, arms in SLATES.items():
        # OLS4 is the accuracy comparator. CV stays the risk reference.
        take, info = base.solve(model, 'ols4', arms, seconds=4.)
        info.update(verified_witness=False, witness_positive=False)
        if take is not None:
            forecast = cv.copy()
            added = 0
            for i in take:
                a, q = model['arm'][i], model['query'][i]
                used = slots[a, q] >= 0
                forecast[slots[a, q, used]] = paths[a, q, used]
                added += int(a >= 4)
            risk = base.policy(forecast, cv, cv, y, valid, np.full(len(cv), recording),
                               easy_cut=easy_cut, hard_cut=hard_cut)['recordings'][recording]
            mass = base.comparison_masses(ols, forecast, cv, y, valid, model['weights'])
            lower = mass['known_delta'] - mass['unknown_radius']
            np.testing.assert_allclose(info['witness_gain_mass'], lower, atol=1e-7, rtol=1e-10)
            if risk['supported']:
                info.update(verified_witness=True, witness_positive=bool(lower > 0.),
                            added_queries_in_witness=added)
            else:
                info.update(status='unresolved_original_risk_postcheck', optimum_proved=False)
        outcomes[slate] = info
    return dict(outcomes=outcomes, reference_mass=model['reference_mass'],
                extra_options=[int((a >= 0).any(axis=-1).sum()) for a in (slots[4:8], slots[8:])])
