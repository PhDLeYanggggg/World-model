"""Finite-completion bounds for fixed causal actions and native partial-label ADE.

Known partial-label costs stay fixed. A wholly unknown row may acquire any
nonempty label mask. These bounds do not calibrate transfer or population risk.
"""
import numpy as np

BUDGET = .02


def completion_bounds(target, action, envelope):
    """Offline diagnostic only; never creates or edits an inference action."""
    y, take, env = np.asarray(target, float), np.asarray(action), np.asarray(envelope, float)
    if (y.ndim != 2 or y.shape[1] != 5 or take.shape != (len(y),)
            or take.dtype != bool or env.shape != (len(y),)
            or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Five moments, fixed Boolean actions and causal max-disagreement required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known] < 0).any():
        raise ValueError('Known nonnegative moments or wholly unknown rows required')
    b, h, r, er, eh = y[known].T
    is_easy = er > 0
    if ((b*h != 0).any() or (b > r).any()
            or ((er != 0) & (er != r)).any()
            or ((eh != 0) & (eh != h)).any()
            or (eh[is_easy] != h[is_easy]).any()
            or (eh[(~is_easy) & (r > 0)] != 0).any()
            or (b+h > env[known]+1e-5).any()):
        raise ValueError('Inconsistent benefit/harm/reference/event/envelope contract')
    chosen = take & known
    unknown_mass = float(env[take & ~known].sum())
    benefit = float(y[chosen, 0].sum())
    harm = float(y[chosen, 1].sum())
    selected_ref = float(y[chosen, 2].sum())
    selected_easy_ref = float(y[chosen, 3].sum())
    easy_harm = float(y[chosen, 4].sum())
    # If floor ADE is zero, benefit must be zero, even if its easy event is unknown.
    easy_benefit = float(y[chosen & (y[:, 3] > 0), 0].sum())
    easy_ref = float(y[known, 3].sum())
    net_lower = benefit-harm-unknown_mass
    easy_net_upper = easy_harm-easy_benefit+unknown_mass
    totals = [unknown_mass, benefit, harm, selected_ref, selected_easy_ref,
              easy_harm, easy_benefit, easy_ref, net_lower, easy_net_upper]
    if not np.isfinite(totals).all():
        raise ValueError('Nonfinite aggregate mass; do not report a bound')
    all_risk = (harm+unknown_mass)/selected_ref if selected_ref > 0 else None
    easy_risk = (easy_harm+unknown_mass)/selected_easy_ref if selected_easy_ref > 0 else None
    # Unknown unselected rows may also enlarge the full easy denominator. A
    # negative numerator divided by a lower denominator is not an upper bound.
    easy_upper_numerator = max(easy_net_upper, 0.) if (~known).any() else easy_net_upper
    easy_upper = easy_upper_numerator/easy_ref if easy_ref > 0 else None
    reasons = []
    if all_risk is None or easy_risk is None:
        reasons.append('selected_reference_undefined')
    if all_risk is not None and all_risk > BUDGET+1e-12:
        reasons.append('all_positive_harm_upper_exceeds_budget')
    if easy_risk is not None and easy_risk > BUDGET+1e-12:
        reasons.append('easy_positive_harm_upper_exceeds_budget')
    if net_lower <= 0:
        reasons.append('positive_completion_utility_not_supported')
    if easy_upper is None or easy_upper > BUDGET+1e-12:
        reasons.append('easy_preservation_not_supported')
    return dict(
        known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        selected_count=int(take.sum()), selected_unknown=int((take & ~known).sum()),
        selected_unknown_envelope_mass=unknown_mass,
        selected_known_benefit_mass=benefit, selected_known_harm_mass=harm,
        selected_known_reference_mass=selected_ref,
        selected_known_easy_reference_mass=selected_easy_ref,
        selected_known_easy_harm_mass=easy_harm,
        selected_known_easy_benefit_mass=easy_benefit,
        full_known_easy_reference_mass=easy_ref,
        all_budget_slack_mass=BUDGET*selected_ref-harm,
        easy_budget_slack_mass=BUDGET*selected_easy_ref-easy_harm,
        selected_net_gain_lower_mass=net_lower,
        all_selected_risk_upper=all_risk, easy_selected_risk_upper=easy_risk,
        easy_degradation_upper=easy_upper,
        finite_completion_supported=not reasons, reasons=reasons,
        known_partial_label_costs_held_fixed=True,
        per_row_label_inference_gate=False, population_safety_guarantee=False)
