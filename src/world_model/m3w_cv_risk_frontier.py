"""Generate causal scene options by solving several predicted risk budgets."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


# Slacks are fractions of predicted query masses, not evaluation thresholds.
GRID = ((0., 0., .02, 0.), (0., 0., .02, .05), (.02, .02, .02, .05),
        (.08, .02, .05, .05), (.02, .08, .05, .05), (.15, .15, .15, 0.))


def round_support(x, offsets, quality, budget, limit=256):
    """Keep fixed rows; search only the fractional rows of the LP vertex."""
    options = []
    fixed = np.array([a + x[a:b].argmax() for a, b in zip(offsets[:-1], offsets[1:], strict=True)])
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        at = a + np.flatnonzero(x[a:b] > 1e-7)
        if len(at) > 1:
            order = np.argsort(-x[at], kind='stable')
            options.append((i, at[order]))
    choices = np.zeros((1, 0), int)
    values = (quality[fixed].sum(0) -
              quality[fixed[[i for i, _ in options]]].sum(0))[None]

    def violation(z):
        return (z[..., 2:] - budget).clip(0.).sum(-1)

    for _, at in options:
        z = (values[:, None] + quality[at][None]).reshape(-1, quality.shape[1])
        ids = np.repeat(choices, len(at), axis=0)
        ids = np.column_stack((ids, np.tile(at, len(choices))))
        if len(z) > limit:
            keep = np.lexsort((-z[:, :2].min(1), violation(z)))[:limit]
            z, ids = z[keep], ids[keep]
        values, choices = z, ids
    bad = violation(values)
    utility = values[:, :2].min(1)
    best = int(np.lexsort((-utility, bad))[0])
    eligible = np.flatnonzero(bad <= 1e-8)
    good = int(eligible[utility[eligible].argmax()]) if len(eligible) else best
    highest = int(utility.argmax())
    out = []
    for k in dict.fromkeys((good, highest)):
        indices = fixed.copy()
        indices[[i for i, _ in options]] = choices[k]
        out.append(indices)
    return out


def generate(gain, risk, legal, reference, scale, grid=GRID):
    """No future labels: solve over frozen predicted increments and masses."""
    if not legal.any():
        return []
    n = len(legal)
    row, action = np.nonzero(np.column_stack((np.ones(n, bool), legal)))
    active = action > 0
    quality = np.zeros((len(row), 6))
    quality[active, :2] = gain[row[active], action[active] - 1] / scale
    quality[active, 2:5] = risk[row[active], action[active] - 1] / scale
    quality[active, 5] = -reference[row[active], 1] / scale
    all_ref, easy_ref = reference.sum(0) / scale
    offsets = np.r_[0, np.bincount(row, minlength=n).cumsum()]
    one = sparse.coo_matrix((np.ones(len(row)), (row, np.arange(len(row)))),
                            shape=(n, len(row) + 1)).tocsr()
    mat = np.column_stack((np.vstack((-quality[:, :2].T, quality[:, 2:].T)), [1., 1., 0., 0., 0., 0.]))
    obj = np.r_[np.zeros(len(row)), -1.]
    bounds = [(0., 1.)] * len(row) + [(None, None)]
    out = {}
    for a, e, s, support in grid:
        budget = np.array([a * all_ref, e * easy_ref, s * easy_ref, -support * easy_ref])
        res = linprog(obj, A_ub=mat, b_ub=np.r_[0., 0., budget], A_eq=one,
            b_eq=np.ones(n), bounds=bounds, method='highs')
        if res.status == 2:
            continue
        if not res.success:
            raise RuntimeError(f'Query LP status {res.status}: {res.message}')
        for z in round_support(res.x[:-1], offsets, quality, budget):
            picked = (action[z] - 1).astype(np.int16)
            out[picked.tobytes()] = picked
    return list(out.values())
