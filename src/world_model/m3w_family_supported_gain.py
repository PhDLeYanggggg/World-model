"""Rank agents within forecast families by supported baseline-relative gain."""
import numpy as np

from src.world_model.m3w_easy_joint_anchor import predict
from src.world_model.m3w_record_joint_actions import coefficients


def moments(raw, atoms, legal, scale, crowd):
    ref, easy, harm, easy_harm, _, utility = atoms.transpose(2, 0, 1)
    supported = (easy > 0.) & (harm <= .02 * ref) & (utility > 0.)
    reward = np.where(supported, utility, 0.) / scale[:, None]
    cost = (np.maximum(harm - .02 * ref, 0.) +
            np.maximum(easy_harm - .02 * easy, 0.)) / scale[:, None]
    weight = legal / crowd[:, None]
    weight /= np.maximum(weight.sum(0), 1e-12)
    g = np.einsum('nfd,nf,nfe->fde', raw, weight, raw, optimize=True)
    y = np.stack((reward, cost), axis=-1)
    b = np.einsum('nfd,nf,nft->fdt', raw, weight, y, optimize=True)
    return g, b


def rank(raw, coef):
    reward, cost = np.einsum('nfd,fdt->nft', raw, coef).transpose(2, 0, 1)
    return np.maximum(reward, 0.) - np.maximum(cost, 0.)


def replace(examples, x, scale, utility, all_cost, easy_cost, legal, old_rank,
            new_rank, bank, floor):
    """Keep the old ordered family slots, including any duplicate families."""
    joint, marginal = predict(x, old_rank)
    radius = np.linalg.norm(bank - floor[:, None], axis=-1).max(2) / scale[:, None]
    values = rank(x, new_rank)
    out = []
    for q, before, families, old in examples:
        used = set()
        rows = []
        for f in families:
            avail = [i for i in q if legal[i, f] and (i, f) not in used]
            if not avail:
                raise ValueError('Cannot preserve original family quotas')
            i = max(avail, key=lambda j: values[j, f])
            rows.append(i)
            used.add((i, f))
        i, j = np.asarray(rows), families
        score = joint[i, j]
        features = np.column_stack((np.ones(len(i)), score, marginal[i, j],
            np.arcsinh(utility[i, j]/scale[i]), np.arcsinh(all_cost[i, j]/scale[i]),
            np.arcsinh(easy_cost[i, j]/scale[i]), np.log1p(radius[i, j]), np.log1p(scale[i]),
            old[:, 8:10], score-score[0], old[:, 11:]))
        out.append((q, i, j, features))
    return out
