"""Past-window motion errors for selecting within a frozen shortlist."""
import numpy as np


def backcast(history, scale, bank, floor, rows, families):
    """Backtest short CV forecasts entirely inside eight observed positions."""
    if history.shape[1:] != (8, 2) or (scale <= 0).any():
        raise ValueError('Require eight observed positions and positive scales')
    v = np.diff(history[rows], axis=1)
    s = scale[rows, None, None]
    one = (v[:, 1:] - v[:, :-1]) / s
    two = (v[:, 2:] + v[:, 1:-1] - 2 * v[:, :-2]) / s
    smooth = (v[:, 3:] - (v[:, :-3] + v[:, 1:-2] + v[:, 2:-1]) / 3) / s
    d = bank[rows, families, 2] - floor[rows, 2]
    d /= np.linalg.norm(d, axis=1).clip(1e-8)[:, None]
    across = np.column_stack((-d[:, 1], d[:, 0]))
    out = []
    for err in (one, two, smooth):
        recent = err[:, -2:].mean(1)
        out.extend((np.log1p(np.linalg.norm(err, axis=2).mean(1)),
                    np.arcsinh((recent * d).sum(1)),
                    np.arcsinh((recent * across).sum(1))))
    return np.column_stack(out)


def augment(examples, scope, history, scale, bank, floor, resolved):
    if scope not in ('backcast', 'resolved'):
        raise ValueError('Unknown history ablation')
    return [
        (q, i, j, np.column_stack((x, backcast(history, scale, bank, floor, i, j)
                                  if scope == 'backcast' else resolved[i, j])))
        for q, i, j, x in examples
    ]
