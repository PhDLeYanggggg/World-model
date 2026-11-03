"""Post-readout cost reliability, with labels used for diagnosis only."""
import numpy as np


def ratio(a, b):
    return float(a/b) if b > 0 else None


def selected_costs(predicted, truth, known, choice, groups):
    p, y, observed, c = map(np.asarray, (predicted, truth, known, choice))
    n = len(p)
    if (p.shape != (n, 7, 5) or y.shape != p.shape or observed.shape != (n,)
            or observed.dtype != bool or c.shape != (n,) or c.dtype.kind not in 'iu'
            or (c < -1).any() or (c >= 7).any()
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or (p < 0).any() or (y < 0).any()):
        raise ValueError('Aligned finite cost moments, separate labels and causal choices required')
    covered = np.concatenate(groups) if groups else np.empty(0, int)
    if not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError('Exactly one causal query per row required')
    take = c >= 0
    rows = np.flatnonzero(take)
    actual, estimate = np.zeros((n, 5)), np.zeros((n, 5))
    actual[rows] = y[rows, c[rows]]
    estimate[rows] = p[rows, c[rows]]
    a, b = actual.sum(0), estimate.sum(0)
    ay, by = actual[observed].sum(0), estimate[observed].sum(0)
    violations = {}
    for name, h, r in (('all', 1, 2), ('easy', 4, 3)):
        pred_bad = known_bad = unsupported = 0
        for at in groups:
            ep, ey = estimate[at].sum(0), actual[at][observed[at]].sum(0)
            pred_bad += int(ep[h] > .02*ep[r])
            known_bad += int(ey[r] > 0 and ey[h] > .02*ey[r])
            unsupported += int(ey[r] <= 0)
        violations[name] = dict(predicted_budget_violating_queries=pred_bad,
            known_realized_budget_violating_queries=known_bad,
            known_reference_unsupported_queries=unsupported)
    return dict(selected=int(take.sum()), selected_known=int((take & observed).sum()),
        selected_unknown=int((take & ~observed).sum()),
        predicted_full_selected_moments=b.tolist(), bounded_label_selected_moments=a.tolist(),
        predicted_on_selected_known_moments=by.tolist(), realized_selected_known_moments=ay.tolist(),
        predicted_full_all_risk=ratio(b[1], b[2]), predicted_full_easy_risk=ratio(b[4], b[3]),
        known_realized_all_risk=ratio(ay[1], ay[2]), known_realized_easy_risk=ratio(ay[4], ay[3]),
        completion_all_risk_upper=ratio(a[1], a[2]), completion_easy_risk_upper=ratio(a[4], a[3]),
        known_harm_mass_over_predicted_same_rows=ratio(ay[1], by[1]),
        known_easy_harm_mass_over_predicted_same_rows=ratio(ay[4], by[4]),
        violations=violations, queries=len(groups),
        labels_used_for_selection=False, post_readout_diagnostic_not_new_safety_gate=True)
