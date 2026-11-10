"""Choose a rotation threshold on grouped calibration readouts only."""

import numpy as np


MARGINS = (0., .001, .0025, .005, .01, .02, .04, .08, None)


def select_threshold(calibration, stable=False):
    if len(calibration) < 2 or any(len(rows) != len(MARGINS) for rows in calibration):
        raise ValueError('Require all registered candidate thresholds for each calibration source')
    reference = [rows[-1] for rows in calibration]
    choices = []
    for i, margin in enumerate(MARGINS[:-1]):
        rows = [r[i] for r in calibration]
        gain = [r['gain_pp'] for r in rows]
        if not np.isfinite(gain).all():
            raise ValueError('Unestimable calibration utility')
        active = sum(r['rotated_edits'] > 0 for r in rows)
        improvement = float(np.mean(gain))
        if (improvement <= 0. or not active or
                sum(r['dual'] for r in rows) < sum(r['dual'] for r in reference)):
            continue
        if stable and (active < 2 or min(gain) < -1e-12):
            continue
        choices.append((improvement, -sum(r['rotated_edits'] for r in rows), margin))
    return max(choices)[-1] if choices else None


def indices(best, score, margin):
    if margin is None:
        return np.zeros_like(best)
    return np.where(score > margin, best+1, 0)
