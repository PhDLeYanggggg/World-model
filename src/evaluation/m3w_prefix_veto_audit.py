"""Read-only diagnosis of fixed terminal nominations and all-prefix vetoes."""
import numpy as np


def diagnose(score, costs, available, nominated, guarded):
    n = len(score)
    if (score.shape != (n, 12, 2) or costs.shape != score.shape or available.shape != (n, 12)
            or any(x.dtype != bool for x in (available, nominated, guarded))
            or nominated.shape != (n,) or guarded.shape != (n,)
            or not np.isfinite(score).all() or not np.isfinite(costs[available]).all()
            or (score < 0).any() or (costs[available] < 0).any()
            or np.any(guarded & ~nominated)
            or np.any(available[:, 1:] & ~available[:, :-1])):
        raise ValueError('Aligned frozen choices and explicit contiguous prefix-label support required')
    fail = score[..., 1] > .1*score[..., 0]
    if not np.array_equal(guarded, nominated & ~fail.any(1)):
        raise ValueError('Choices do not follow the registered prefix guard')
    vetoed = nominated & ~guarded
    full = available[:, -1]
    first = np.argmax(fail, 1) + 1
    result = dict(nominated=int(nominated.sum()), guarded=int(guarded.sum()), vetoed=int(vetoed.sum()),
                  first_veto_prefix_counts={str(k): int((vetoed & (first == k)).sum()) for k in range(1, 13)},
                  groups={}, prefix_quality=[])
    for name, mask in (('retained', guarded), ('vetoed', vetoed)):
        ok = mask & full
        terminal_benefit = ok & (costs[:, -1, 0] > 0)
        any_harm = np.any(np.where(available, costs[..., 1], 0) > 0, 1)
        result['groups'][name] = dict(rows=int(mask.sum()), complete=int(ok.sum()),
            incomplete=int((mask & ~full).sum()), no_valid_prefix=int((mask & ~available.any(1)).sum()),
            complete_terminal_beneficial=int(terminal_benefit.sum()),
            complete_terminal_harmful=int((ok & (costs[:, -1, 1] > 0)).sum()),
            complete_any_prefix_harm=int((ok & any_harm).sum()),
            complete_terminal_beneficial_but_prefix_harmed=int((terminal_benefit & any_harm).sum()))
    use = nominated & full
    for k in range(12):
        row = dict(prefix=k+1, complete_nominated=int(use.sum()))
        if use.any():
            row.update(predicted_harm=float(score[use, k, 1].mean()), realized_harm=float(costs[use, k, 1].mean()),
                       predicted_benefit=float(score[use, k, 0].mean()), realized_benefit=float(costs[use, k, 0].mean()),
                       MSE=float(np.square(score[use, k]-costs[use, k]).mean()))
        result['prefix_quality'].append(row)
    return result
