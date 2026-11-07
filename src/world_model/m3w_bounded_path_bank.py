"""Bounded alternatives to a full-path substitution."""
import numpy as np

LEVELS = (1., .5, .25)


def blend(bank, floor, level):
    if level not in LEVELS:
        raise ValueError('Use a registered amplitude')
    return bank if level == 1. else floor[:,None] + level*(bank-floor[:,None])


def selected(bank, floor, choice):
    """Decode level-major choices. An unselected row is exactly the floor."""
    n = bank.shape[1]
    if choice.shape != (len(floor),) or ((choice < -1) | (choice >= n*len(LEVELS))).any():
        raise ValueError('Invalid bounded-bank choice')
    out = floor.copy()
    for i, level in enumerate(LEVELS):
        rows = np.flatnonzero((choice >= i*n) & (choice < (i+1)*n))
        if len(rows):
            a = bank[rows,choice[rows]-i*n]
            out[rows] = a if i==0 else floor[rows]+level*(a-floor[rows])
    return out


def attenuate(choice, width, level):
    if ((choice < -1) | (choice >= width)).any():
        raise ValueError('Only original choices may be attenuated')
    return np.where(choice>=0,choice+width*LEVELS.index(level),-1)
