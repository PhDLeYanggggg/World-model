"""Interaction-aware angle choice with at most one altered direction."""

import numpy as np


def proxy_gains(bank, anchor, references, history, recipients, scale):
    """Worst gains over causal surrogates, without using future labels."""
    if len(bank) != 2 or len(recipients) != 2 or len(set(recipients)) != 2:
        raise ValueError('Expected the two original recipients')
    individual, relative = [], []
    for i, row in enumerate(recipients):
        p, before = bank[i, 1:], anchor[row]
        ref = references[:, row]
        err0 = np.linalg.norm(before-ref, axis=-1).mean(1)
        err = np.linalg.norm(p[:, None]-ref, axis=-1).mean(2)
        individual.append(((err0-err)/scale[row]).min(1))
        others = np.flatnonzero(np.arange(len(anchor)) != row)
        distance = np.linalg.norm(history[others, -1]-history[row, -1], axis=-1)
        w = 1./np.maximum(distance, scale[row])
        w /= w.sum()
        rel0 = before-anchor[others]
        target = ref[:, None]-references[:, others]
        rel = p[:, None]-anchor[others]
        r0 = np.linalg.norm(rel0-target, axis=-1).mean(-1)@w
        r = np.linalg.norm(rel[:, None]-target, axis=-1).mean(-1)@w
        relative.append(((r0-r)/scale[row]).min(1))
    return np.asarray(individual), np.asarray(relative)


def choose(individual, relative):
    if individual.shape != relative.shape or individual.shape[0] != 2:
        raise ValueError('Expected matched individual and interaction scores')

    def best(z):
        row, angle = np.unravel_index(z.argmax(), z.shape)
        return (int(row), int(angle)+1) if z[row, angle] > 0. else None

    a = best(individual)
    b = best(np.where((individual > 0.) & (relative > 0.), individual+relative, -np.inf))
    return dict(independent=a, interaction=b,
                independent_at_interaction=None if b is None else (b[0], int(individual[b[0]].argmax())+1))
