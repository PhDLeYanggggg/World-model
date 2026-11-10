"""Same-count donor neighbors that satisfy the original additive risk terms."""

import numpy as np

from src.evaluation.m3w_count_motion_capacity import respects_tokens, within_cap


def neighbors(good, movement, cap, terms):
    """Propose low- and high-utility risk-feasible one-edit exchanges."""
    gain, risk, support, legal, offset, easy_cap, mass = terms
    if mass <= 0.:
        return []
    q = np.arange(len(good))
    now_r = risk[q, good].sum(0)
    now_g = offset + gain[q, good].sum(0)
    now_easy = support[q, good].astype(int)
    now_move = movement[q, good].sum()
    destinations = q[(good < 2) & legal[q, np.minimum(good + 1, 2)]]
    choices = []
    for src in np.flatnonzero(good):
        dst = destinations[destinations != src]
        a, b = good[src], good[dst]
        move = now_move + movement[src, a - 1] - movement[src, a] + movement[dst, b + 1] - movement[dst, b]
        r = now_r + risk[src, a - 1] - risk[src, a] + risk[dst, b + 1] - risk[dst, b]
        e = now_easy.sum() + int(support[src, a - 1]) - now_easy[src] + support[dst, b + 1] - now_easy[dst]
        g = now_g + gain[src, a - 1] - gain[src, a] + gain[dst, b + 1] - gain[dst, b]
        v = 100. * g.min(axis=1) / mass
        good_value = 100. * now_g.min() / mass
        valid = within_cap(move, cap) & (e > 0) & (r <= np.array([0., 0., easy_cap]) + 1e-12).all(1)
        valid &= v < good_value - 1e-6
        for j in np.flatnonzero(valid):
            plan = good.copy()
            plan[src] -= 1
            plan[dst[j]] += 1
            if within_cap(movement[q, plan].sum(), cap) and respects_tokens(plan, legal[:, 1]):
                choices.append((float(v[j]), int(src), int(dst[j]), plan))
    if not choices:
        return []
    choices.sort(key=lambda x: x[:3])
    return [choices[0][3]] if len(choices) == 1 else [choices[0][3], choices[-1][3]]
