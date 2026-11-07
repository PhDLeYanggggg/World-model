"""Matched multi-action allocation from past-only paths and fitted cost moments."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors

ARMS = ('floor', 'independent', 'scene', 'pair_additive', 'joint', 'budgeted_joint')
PAIR_WEIGHT = .10


def descriptors(history, paths, floor, crowd):
    v = np.diff(history, axis=1)
    speed = np.linalg.norm(v, axis=-1)
    scale = 1. + 6.5 * np.median(speed, axis=1)
    direction = v[:, -3:].mean(1)
    direction /= np.linalg.norm(direction, axis=-1).clip(1e-8)[:, None]
    d = (paths - floor[:, None]) / scale[:, None, None, None]
    radius = np.linalg.norm(d, axis=-1).max(2)
    summary = np.column_stack((np.log1p(speed), np.log1p(np.linalg.norm(np.diff(v, axis=1), axis=-1)),
                               np.log1p(crowd), np.log1p(scale)))
    z = np.concatenate((np.broadcast_to(summary[:, None], (*paths.shape[:2], summary.shape[1])),
        np.log1p(radius)[..., None], np.log1p(np.linalg.norm(d[:, :, [2, 5, 11]], axis=-1)),
        np.arcsinh((d[:, :, [2, 5, 11]] * direction[:, None, None]).sum(-1)),
        np.log1p(np.linalg.norm(paths[:, :, [2, 5, 11]] / scale[:, None, None, None], axis=-1))), axis=-1)
    z = np.clip(z, -12., 12.)
    return np.concatenate((np.ones((*z.shape[:2], 1)), z, np.tanh(z)), axis=-1), scale


def fitting_moments(x, paths, floor, target, valid, scale, weights, easy_cut):
    ref, _ = errors(floor, target, valid)
    known = valid.any(1)
    ref = np.where(known, ref, 0.)
    easy = known & (ref > 0) & (ref <= easy_cut)
    w = weights / weights.sum()
    gram, rhs = [], []
    for a in range(paths.shape[1]):
        e, _ = errors(paths[:, a], target, valid)
        gain = np.where(known, ref - e, 0.)
        harm = np.maximum(-gain, 0.)
        y = np.column_stack((gain / scale, harm / scale, easy * harm / scale,
                             ref / scale, easy * ref / scale, ~known))
        z = x[:, a]
        gram.append(z.T @ (w[:, None] * z)); rhs.append(z.T @ (w[:, None] * y))
    return np.stack(gram), np.stack(rhs)


def coefficients(fitting_sources):
    g, b = (np.mean([m[j] for m in fitting_sources], axis=0) for j in (0, 1))
    ridge = np.eye(g.shape[-1]) * .002
    ridge[0, 0] = 0.
    return np.linalg.solve(g + ridge, b)


def decision_costs(x, coef, paths, floor, scale):
    d = paths - floor[:, None]
    radius = np.linalg.norm(d, axis=-1).max(2)
    p = np.einsum('nad,adc->nac', x, coef)
    gain, harm, easy_harm, ref, easy_ref, unknown = p.transpose(2, 0, 1)
    missing = unknown.clip(0., 1.) * radius
    harm = np.minimum(harm.clip(0.) * scale[:, None], radius - missing)
    easy_harm = np.minimum(easy_harm.clip(0.) * scale[:, None], harm)
    ref = ref.clip(0.) * scale[:, None]
    easy_ref = np.minimum(easy_ref.clip(0.) * scale[:, None], ref)
    gain = np.clip(gain * scale[:, None], -harm, radius - missing - harm)
    utility = gain - missing
    # These are estimates for choosing actions, not the original scoring gate.
    return utility, harm + missing - .02 * ref, easy_harm + missing - .02 * easy_ref, radius > 0


def local_edges(history):
    origin = history[:, -1]
    dist = np.linalg.norm(origin[:, None] - origin[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    near = np.argsort(dist, axis=1, kind='stable')[:, :min(4, len(origin)-1)]
    i = np.repeat(np.arange(len(origin)), near.shape[1]); j = near.ravel()
    pairs = np.unique(np.column_stack((np.minimum(i, j), np.maximum(i, j))), axis=0)
    return pairs.T


def choose_query(history, floor, paths, ols, scale, utility, legal):
    """All four matched arms really intervene on the same number of rows."""
    n = len(floor)
    result = np.full((len(ARMS)-1, n), -1, dtype=np.int16)
    k = min(2, n-1, int(legal.sum(0).max()))
    if not k:
        return result
    score = np.where(legal, utility / scale[:, None], -np.inf)
    best = score.argmax(1); value = score[np.arange(n), best]
    take = np.argsort(-value, kind='stable')[:k]
    result[0, take] = best[take]
    same = np.argsort(-score, axis=0, kind='stable')[:k]
    action = score[same, np.arange(score.shape[1])].sum(0).argmax()
    result[1, same[:, action]] = action

    i, j = local_edges(history)
    degree = np.bincount(np.r_[i, j], minlength=n)
    pair_scale = np.sqrt(scale[i] * scale[j])[:, None, None]
    weight = 1. / np.sqrt(degree[i] * degree[j])
    d = paths[np.arange(n), best] - floor
    e = ((floor[i] - floor[j]) - (ols[i] - ols[j])) / pair_scale
    di, dj = d[i] / pair_scale, d[j] / pair_scale
    marginal = np.zeros(n)
    np.add.at(marginal, i, weight * (2 * e * di + di**2).sum(-1).mean(1))
    np.add.at(marginal, j, weight * (-2 * e * dj + dj**2).sum(-1).mean(1))
    adjusted = value - PAIR_WEIGHT * marginal
    take = np.argsort(-adjusted, kind='stable')[:k]
    result[2, take] = best[take]
    if k == 2:
        joint = adjusted[:, None] + adjusted[None]
        joint[i, j] += 2 * PAIR_WEIGHT * weight * (di * dj).sum(-1).mean(1)
        joint[np.tril_indices(n)] = -np.inf
        take = np.asarray(np.unravel_index(joint.argmax(), joint.shape))
    result[3, take] = best[take]
    result[4] = result[3]
    return result


def budget_prefix(choice, utility, all_excess, easy_excess, scale):
    """Select a positive prefix under predicted recording constraints or abstain."""
    take = np.flatnonzero(choice >= 0)
    idx = (take, choice[take]); u = utility[idx]
    keep = u > 0.
    take = take[keep]; u = u[keep]
    order = np.argsort(-u / scale[take], kind='stable'); take = take[order]
    idx = (take, choice[take])
    good = np.flatnonzero((np.cumsum(all_excess[idx]) <= 0.) & (np.cumsum(easy_excess[idx]) <= 0.))
    out = np.full_like(choice, -1)
    if len(good):
        use = take[:good[-1]+1]
        out[use] = choice[use]
    return out


def selected_paths(choice, paths, floor):
    return np.where(choice[..., None, None] >= 0, paths[np.arange(len(floor)), choice.clip(0)], floor)


def pair_error(history, prediction, target, valid):
    i, j = local_edges(history)
    mask = valid[i] & valid[j]
    n = int(mask.sum())
    err = (prediction[i] - prediction[j]) - (target[i] - target[j])
    return float(np.linalg.norm(err[mask], axis=-1).sum() / n) if n else None, n
