"""Evaluation-only arithmetic witnesses on an unchanged causal selection set."""
import numpy as np

BUDGET = .02
WITNESSES = ('numerator_only', 'denominator_only',
             'both_single_replacements_fail', 'only_joint_replacement_fails')


def _channel(y, p, harm, reference):
    h, r = y[:, harm], y[:, reference]
    ph, pr = p[:, harm], p[:, reference]
    observed, predicted = h - BUDGET * r, ph - BUDGET * pr
    numerator, denominator = h - ph, BUDGET * (pr - r)
    false = observed > 1e-12
    replace_h, replace_r = h - BUDGET * pr > 1e-12, ph - BUDGET * r > 1e-12
    masks = (false & replace_h & ~replace_r, false & ~replace_h & replace_r,
             false & replace_h & replace_r, false & ~replace_h & ~replace_r)
    if not np.array_equal(np.sum(masks, axis=0), false.astype(int)):
        raise ValueError('Arithmetic witnesses must partition false admissions')
    residual = float(np.sum(observed - predicted - numerator - denominator))
    if not np.allclose(observed - predicted, numerator + denominator, rtol=1e-10, atol=1e-8):
        raise ValueError('Selected score decomposition failed')
    return dict(false_admitted_known_rows=int(false.sum()),
        witness_counts={key: int(mask.sum()) for key, mask in zip(WITNESSES, masks)},
        witness_true_excess_mass={key: float(observed[mask].sum()) for key, mask in zip(WITNESSES, masks)},
        observed_harm_mass=float(h.sum()), predicted_harm_mass=float(ph.sum()),
        observed_reference_mass=float(r.sum()), predicted_reference_mass=float(pr.sum()),
        observed_score_mass=float(observed.sum()), predicted_score_mass=float(predicted.sum()),
        numerator_optimism_mass=float(numerator.sum()),
        denominator_optimism_mass=float(denominator.sum()),
        positive_numerator_optimism_mass=float(np.maximum(numerator, 0).sum()),
        positive_denominator_optimism_mass=float(np.maximum(denominator, 0).sum()),
        decomposition_residual=residual)


def _group(y, p, choices, cv, floor, known, envelope, easy_cut):
    selected = choices >= 0
    rows = np.flatnonzero(selected & known)
    sy, sp = y[rows, choices[rows]], p[rows, choices[rows]]
    easy = known & (cv > 0) & (cv <= easy_cut)
    se = easy[rows]
    unknown = np.flatnonzero(selected & ~known)
    out = dict(rows=len(y), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        selected_known_rows=len(rows), selected_unknown_rows=len(unknown),
        selected_unknown_envelope_mass=float(envelope[unknown, choices[unknown]].sum()),
        full_known_reference_mass=float(floor[known].sum()),
        channels={name: _channel(sy, sp, h, r) for name, h, r in (('all', 1, 2), ('easy', 4, 3))},
        easy_support=dict(full_known_easy_rows=int(easy.sum()),
            full_known_easy_reference_mass=float(floor[easy].sum()),
            selected_known_easy_rows=int(se.sum()),
            selected_known_easy_zero_reference_rows=int((se & (sy[:, 2] == 0)).sum()),
            selected_known_noneasy_rows=int((~se).sum()),
            selected_easy_reference_mass=float(sy[:, 3].sum()),
            predicted_easy_reference_on_known_easy=float(sp[se, 3].sum()),
            predicted_easy_reference_on_known_noneasy=float(sp[~se, 3].sum())),
        selected_net_benefit_mass=float((sy[:, 0] - sy[:, 1]).sum()),
        predicted_selected_net_benefit_mass=float((sp[:, 0] - sp[:, 1]).sum()),
        selected_nonpositive_benefit_rows=int((sy[:, 0] - sy[:, 1] <= 0).sum()))
    return out


def analyze(packet, prediction, choices, *, easy_cut):
    """Labels are read only after frozen causal choices have been supplied."""
    y, p, c = np.asarray(packet['y']), np.asarray(prediction, dtype=np.float64), np.asarray(choices)
    n = len(packet['ids'])
    if (y.shape != (n, 7, 5) or p.shape != y.shape or c.shape != (n,)
            or not np.issubdtype(c.dtype, np.integer) or np.any((c < -1) | (c > 6))
            or not np.isfinite(p).all() or np.any(p < 0) or not np.isfinite(easy_cut) or easy_cut < 0):
        raise ValueError('Aligned finite nonnegative moments and legal choices required')
    known = np.isfinite(y).all((1, 2))
    if not np.all(known | np.isnan(y).all((1, 2))):
        raise ValueError('Wholly unknown targets must stay unknown, never partly imputed')
    cv, floor = np.asarray(packet['cv_ade']), np.asarray(packet['floor_ade'])
    env, rec = np.asarray(packet['envelope']), np.asarray(packet['recordings'])
    if (cv.shape != (n,) or floor.shape != (n,) or env.shape != (n, 7) or rec.shape != (n,)
            or not np.array_equal(np.isfinite(cv), known) or not np.array_equal(np.isfinite(floor), known)
            or not np.isfinite(env).all() or np.any(env < 0)):
        raise ValueError('Aligned reference, observation and envelope support required')
    selected = np.flatnonzero(c >= 0)
    chosen = p[selected, c[selected]]
    if (np.any(chosen[:, 1] > BUDGET * chosen[:, 2] + 1e-9)
            or np.any(chosen[:, 4] > BUDGET * chosen[:, 3] + 1e-9)):
        raise ValueError('Selected actions must satisfy the unchanged causal decoder')
    out = _group(y, p, c, cv, floor, known, env, easy_cut)
    out['records'] = {str(r): _group(y[at], p[at], c[at], cv[at], floor[at], known[at], env[at], easy_cut)
        for r in sorted(set(rec)) for at in (rec == r,)}
    out['actions'] = {str(a): _group(y[at], p[at], c[at], cv[at], floor[at], known[at], env[at], easy_cut)
        for a in range(7) for at in (c == a,)}
    return dict(out, future_informed_diagnostic_only=True, new_policy_constructed=False,
        unknown_outcomes_imputed=False, causal_root_cause_claim=False,
        oracle_replacement_available_at_inference=False, risk_budget=BUDGET)
