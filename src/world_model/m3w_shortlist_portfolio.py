"""Eight-slot retrieval from complementary causal cost rankings."""
import numpy as np

from src.world_model.m3w_easy_joint_anchor import predict


def retrieve(joint, gain, all_excess, easy_excess, legal):
    """Round-robin rankings, preferring a new agent whenever possible."""
    valid = np.flatnonzero(legal.ravel())
    n = legal.shape[1]
    j, u = joint.ravel()[valid], gain.ravel()[valid]
    excess = (np.maximum(all_excess, 0.) + np.maximum(easy_excess, 0.)).ravel()[valid]
    rankings = [valid[np.lexsort((-j, -v))] for v in (j, u, u-excess, -excess)]
    picked, seen, agents = [], set(), set()
    for slot in range(min(8, len(valid))):
        order = rankings[slot % len(rankings)]
        idx = next((i for i in order if i//n not in agents), None)
        if idx is None:
            idx = next(i for i in order if i not in seen)
        picked.append(idx)
        seen.add(idx)
        agents.add(idx//n)
    return np.asarray(picked, dtype=int)


def features(raw, scale, utility, all_cost, easy_cost, legal, rank, bank, floor, groups):
    joint, marginal = predict(raw, rank)
    n = legal.shape[1]
    u, a, e = (v / scale[:, None] for v in (utility, all_cost, easy_cost))
    radius = np.linalg.norm(bank - floor[:, None], axis=-1).max(2) / scale[:, None]
    examples = []
    for q in groups:
        if len(q) < 2 or not legal[q].any():
            continue
        take = retrieve(joint[q], u[q], a[q], e[q], legal[q])
        i, j = q[take//n], take%n
        score = joint[i, j]
        x = np.column_stack((np.ones(len(i)), score, marginal[i, j], np.arcsinh(u[i, j]),
            np.arcsinh(a[i, j]), np.arcsinh(e[i, j]), np.log1p(radius[i, j]),
            np.log1p(scale[i]), np.full(len(i), np.log1p(len(q))), np.arange(len(i))/8,
            score - score[0], np.eye(n)[j]))
        examples.append((q, i, j, x))
    return examples


def winners(example, coef):
    _, i, j, x = example
    utility, all_cost, easy_cost, signed, event = (x @ coef).T
    safe_utility = utility - sum(np.maximum(z, 0.) for z in (all_cost, easy_cost, signed))
    return [(float(np.clip(event[k], 0., 1.)), int(i[k]), int(j[k]))
            for k in (safe_utility.argmax(), event.argmax())]


def options(direct, examples, coef):
    arms = [[], []]
    for q, i, j, x in examples:
        if (direct[q] >= 0).any():
            break
        for out, candidate in zip(arms, winners((q, i, j, x), coef)):
            out.append((*candidate, q))
    return arms
