"""Local timing substitutions with the incumbent's recipients and peaks fixed."""

from itertools import product

import numpy as np

from src.world_model import m3w_av2_temporal_menu as timing
from src.world_model import m3w_av2_history_replay as history


PROFILES = np.asarray([(0, 0), *[p for p in product(range(3), repeat=2) if p != (0, 0)]])


def paths(cv, incumbent):
    edited = np.flatnonzero(np.any(incumbent != cv, axis=(1, 2)))
    if len(edited) != 2:
        raise ValueError("Require exactly two incumbent edits")
    variants = timing.expanded(cv[edited], incumbent[edited, None])
    bank = np.broadcast_to(incumbent, (len(PROFILES), *incumbent.shape)).copy()
    bank[:, edited] = variants[np.arange(2)[None], PROFILES]
    actual = np.linalg.norm(bank-cv, axis=-1).max(-1)
    expected = np.linalg.norm(incumbent-cv, axis=-1).max(-1)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(bank[0], incumbent)
    distinct = [0]
    for i, p in enumerate(bank[1:], 1):
        if not np.any(np.all(np.isclose(bank[distinct], p, atol=1e-10, rtol=1e-10), axis=(1, 2, 3))):
            distinct.append(i)
    return bank[distinct], np.broadcast_to(edited, (len(distinct), 2)).copy(), PROFILES[distinct]


def replay_score(probes):
    """Require mean observed trajectory and pair improvements over the incumbent."""
    p = probes.reshape(len(probes), len(history.PREFIXES), 8)
    delta = (p-p[:1]).mean(1)
    return np.minimum(delta[:, 0], delta[:, 5])
