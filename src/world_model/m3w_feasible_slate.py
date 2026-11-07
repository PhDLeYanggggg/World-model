"""One-agent replacements and joint feasibility features on a matched slate."""
import numpy as np

from src.world_model import m3w_native_support_retrieval as retrieval


def options(proposal, utility, scale, legal):
    selected = np.flatnonzero(proposal >= 0)
    if not len(selected):
        return np.empty((0, len(proposal)), dtype=proposal.dtype), np.empty(0, dtype=int)
    u = np.where(legal, utility/scale[:, None], -np.inf)
    replace = selected[u[selected, proposal[selected]].argmin()]
    anchor = selected[selected != replace]
    best = u.argmax(1)
    best[replace] = proposal[replace]
    rank = u[np.arange(len(u)), best]
    eligible = np.flatnonzero(np.isfinite(rank) & ~np.isin(np.arange(len(u)), anchor))
    ordered = eligible[np.argsort(-rank[eligible], kind='stable')[:retrieval.SHORTLIST]]
    if replace not in ordered:
        ordered = np.r_[ordered[:retrieval.SHORTLIST-1], replace]
    ordered = ordered[np.argsort(-rank[ordered], kind='stable')]
    slate = np.repeat(proposal[None], len(ordered), axis=0)
    slate[:, replace] = -1
    slate[np.arange(len(ordered)), ordered] = best[ordered]
    return slate, ordered


def descriptors(slate, history, row_features, bank, floor, scale, u, a, e, reference):
    mask = slate >= 0
    idx = np.where(mask)[1].reshape(len(slate), -1)
    at = idx, slate[mask].reshape(idx.shape)
    normalizer = scale[idx].mean(1)
    cost = np.stack([v[at] for v in (u, a, e, reference)], axis=-1) / normalizer[:, None, None]
    delta = bank[at]-floor[idx]
    radius = np.linalg.norm(delta, axis=-1).max(2)/normalizer[:, None]
    spread = np.linalg.norm(delta[:, 0]-delta[:, -1], axis=-1).max(1)/normalizer
    distance = np.linalg.norm(history[idx[:, 0], -1]-history[idx[:, -1], -1], axis=-1)/normalizer
    return np.column_stack((np.arcsinh(cost.sum(1)), np.arcsinh(cost.min(1)), np.arcsinh(cost.max(1)),
        radius.mean(1), radius.max(1), np.log1p(spread), np.log1p(distance),
        row_features[idx].mean(1), row_features[idx].max(1)))


def choose(probability, gate_legal, gate_active):
    best = int(np.argmax(probability))
    if not gate_active:
        return best, None
    if not gate_legal.any():
        raise ValueError('Frozen active query must retain an eligible option')
    return best, int(np.argmax(np.where(gate_legal, probability, -np.inf)))
