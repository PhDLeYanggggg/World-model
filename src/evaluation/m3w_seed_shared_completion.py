"""Bounds and explicit completions for one unknown label shared by model seeds."""

from itertools import permutations

import numpy as np


def seed_shared_completion(ref, pred, target, valid, cv):
    """Return total 12-grid error gains before division by observed CV mass.

    Three equally weighted seeds refer to the same target at each step.
    Any bijection between their predictions preserves the mean contrast;
    its mean distance bounds the contrast for any unknown position.
    """
    ref, pred, target, valid, cv = map(np.asarray, (ref, pred, target, valid, cv))
    if (ref.shape != (3, *target.shape) or pred.shape != ref.shape or cv.shape != target.shape
            or target.shape != (*valid.shape, 2) or valid.ndim != 2 or valid.dtype != bool
            or not all(np.isfinite(x).all() for x in (ref, pred, cv, target[valid]))):
        raise ValueError("Three aligned seeds, finite paths and a boolean label mask required")

    a, b = ref[:, ~valid], pred[:, ~valid]
    distance = np.linalg.norm(a[:, None] - b[None], axis=-1)
    diagonal = distance[np.arange(3), np.arange(3)].mean(0)
    coupled = np.minimum.reduce([distance[np.arange(3), p].mean(0)
                                 for p in permutations(range(3))])
    known = np.linalg.norm(ref[:, valid]-target[valid], axis=-1)
    known -= np.linalg.norm(pred[:, valid]-target[valid], axis=-1)
    observed = float(known.sum(axis=1).mean())
    old_debit, debit = float(diagonal.sum()), float(coupled.sum())

    # Use a whole template for each row, not a different extreme point
    # for every missing step. At observed steps all templates use the label.
    templates = np.concatenate((cv[None], ref, pred))
    missing = []
    for path in templates:
        diff = np.linalg.norm(a-path[~valid], axis=-1) - np.linalg.norm(b-path[~valid], axis=-1)
        totals = np.zeros(valid.shape, dtype=float)
        totals[~valid] = diff.mean(0)
        missing.append(totals.sum(1))
    low_witness = float(np.min(missing, axis=0).sum())
    high_witness = float(np.max(missing, axis=0).sum())
    if (debit > old_debit+1e-9 or low_witness < -debit-1e-9 or high_witness > debit+1e-9):
        raise ValueError("Completion exceeds certified outer envelope")

    return dict(
        observed=observed, diagonal_lower=observed-old_debit, diagonal_upper=observed+old_debit,
        shared_lower=observed-debit, shared_upper=observed+debit,
        template_low=observed+low_witness, template_high=observed+high_witness,
        debit_reduction=old_debit-debit, missing_steps=int((~valid).sum()),
        affected_missing_steps=int((diagonal > 1e-12).sum()),
        tighter_missing_steps=int((diagonal-coupled > 1e-12).sum()),
    )
