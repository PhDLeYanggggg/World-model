"""Rotate an existing CV-relative edit without changing its displacement budget."""

import numpy as np

from src.world_model import m3w_av2_incumbent_timing as timing


ANGLE_DEG = 15.
MODES = ("timing", "direction", "union")


def rotate(cv, incumbent):
    edited = np.flatnonzero(np.any(incumbent != cv, axis=(1, 2)))
    if len(edited) != 2:
        raise ValueError("Require exactly two edits")
    diff = incumbent[edited]-cv[edited]
    angle = np.deg2rad([0., -ANGLE_DEG, ANGLE_DEG])
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.stack((c, s, -s, c), axis=-1).reshape(3, 2, 2)
    variants = cv[edited, None]+np.einsum("atd,mdk->amtk", diff, matrix)
    variants[:, 0] = incumbent[edited]
    bank = np.broadcast_to(incumbent, (len(timing.PROFILES), *incumbent.shape)).copy()
    bank[:, edited] = variants[np.arange(2)[None], timing.PROFILES]
    actual = np.linalg.norm(bank-cv, axis=-1)
    expected = np.linalg.norm(incumbent-cv, axis=-1)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-9, rtol=1e-9)
    return bank


def menus(cv, incumbent):
    timed, _, _ = timing.paths(cv, incumbent)
    angular = rotate(cv, incumbent)
    if len(timed) != len(angular):
        raise ValueError("Require an exact equal-size timing and rotation comparison")
    bank = np.concatenate((timed, angular[1:]), axis=0)
    slots = np.flatnonzero(np.any(incumbent != cv, axis=(1, 2)))
    return bank, np.broadcast_to(slots, (len(bank), len(slots))).copy(), {
        "timing": np.arange(len(timed)), "direction": np.r_[0, np.arange(len(timed), len(bank))], "union": np.arange(len(bank))}
