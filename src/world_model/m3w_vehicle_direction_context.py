"""Observed geometry and matched causal motion models for an angle repair."""

import numpy as np

from src.world_model import m3w_vehicle_residual as residual


def design(history, offsets, counts):
    """Nearest two neighbors in the shared query frame; ego-aligned features."""
    if history.shape[1:] != (8, 2) or offsets.shape != (len(history), 2):
        raise ValueError('History and shared offsets must align')
    if sum(counts) != len(history) or np.any(np.asarray(counts) <= 0):
        raise ValueError('Invalid query bounds')
    own, cv, rotation, scale = residual.encode(history)
    peers = np.zeros((len(history), 2, 6), dtype=np.float32)
    velocity = np.diff(history[:, -4:], axis=1).mean(1)
    lo = 0
    for n in counts:
        q = slice(lo, lo+n)
        if n > 1:
            xy = offsets[q][None] - offsets[q][:, None]
            dist = np.linalg.norm(xy, axis=-1)
            np.fill_diagonal(dist, np.inf)
            pick = np.argsort(dist, axis=1, kind='stable')[:, :min(n-1, 2)]
            ego = np.arange(n)[:, None]
            pos = (xy[ego, pick] @ rotation[q]) / (8*scale[q, None, None])
            dv = ((velocity[q][pick]-velocity[q][:, None]) @ rotation[q]) / scale[q, None, None]
            raw = np.concatenate((pos, dv, np.log1p(dist[ego, pick])[..., None],
                                  np.ones((*pick.shape, 1))), axis=-1)
            peers[q, :pick.shape[1]] = np.clip(raw, -8., 8.)
        lo += n
    return np.concatenate((own, peers.reshape(len(own), -1)), axis=1), cv, rotation, scale


def ablate(x):
    """Same tensor width and architecture; remove all neighbor signals."""
    out = x.copy()
    out[:, 17:] = 0.
    return out


def matched(individual, chosen):
    if chosen is None:
        return None
    actor, _ = chosen
    return actor, int(individual[actor].argmax())+1
