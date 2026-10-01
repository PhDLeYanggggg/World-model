"""Development-only source-validation policy choice; no row-label inference gate."""
import numpy as np

NAMES = ('mse', 'final', 'initial')
BUDGET = .02


def validation_readout(target, action):
    y, take = np.asarray(target, float), np.asarray(action)
    if y.ndim != 2 or y.shape[1] != 5 or take.shape != (len(y),) or take.dtype != bool:
        raise ValueError('Five supervised moments and boolean causal actions required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known] < 0).any():
        raise ValueError('Targets must be known nonnegative moments or wholly unknown')
    chosen = take & known
    reference = float(y[known, 2].sum())
    selected_ref = float(y[chosen, 2].sum())
    easy_ref = float(y[known, 3].sum())
    selected_easy_ref = float(y[chosen, 3].sum())
    easy = known & (y[:, 3] > 0)
    benefit, harm = y[:, 0], y[:, 1]
    all_risk = float(harm[chosen].sum())/selected_ref if selected_ref > 0 else None
    easy_risk = float(y[chosen, 4].sum())/selected_easy_ref if selected_easy_ref > 0 else None
    gain = float((benefit[chosen]-harm[chosen]).sum())/reference if reference > 0 else None
    easy_degradation = float((harm[chosen & easy]-benefit[chosen & easy]).sum())/easy_ref if easy_ref > 0 else None
    unknown = int(take[~known].sum())
    reasons = []
    if unknown: reasons.append('selected_outcomes_unknown_on_source_validation')
    if all_risk is None or easy_risk is None: reasons.append('selected_reference_undefined')
    if all_risk is not None and all_risk > BUDGET + 1e-12: reasons.append('all_positive_harm_exceeds_budget')
    if easy_risk is not None and easy_risk > BUDGET + 1e-12: reasons.append('easy_positive_harm_exceeds_budget')
    if gain is None or gain <= 0: reasons.append('no_positive_validation_utility')
    if easy_degradation is None or easy_degradation > BUDGET + 1e-12: reasons.append('easy_preservation_not_supported')
    return dict(eligible=not reasons, reasons=reasons, all_gain_fraction=gain,
                all_selected_harm_ratio=all_risk, easy_selected_harm_ratio=easy_risk,
                easy_degradation_fraction=easy_degradation, selected_unknown=unknown,
                selected_count=int(take.sum()), known_rows=int(known.sum()),
                unknown_rows=int((~known).sum()))


def choose(target, actions):
    """Use labels only for source-level model choice, never per-row deployment."""
    if set(actions) != set(NAMES): raise ValueError('All three frozen candidates required')
    rows = {name: validation_readout(target, actions[name]) for name in NAMES}
    eligible = [name for name in NAMES if rows[name]['eligible']]
    selected = min(eligible, key=lambda n: (-rows[n]['all_gain_fraction'], rows[n]['selected_count'], NAMES.index(n))) if eligible else 'fallback'
    return dict(selected=selected, validation=rows,
                status='source_validation_point_supported' if eligible else 'no_supported_nonempty_policy',
                independent_calibration=False, population_safety_guarantee=False)
