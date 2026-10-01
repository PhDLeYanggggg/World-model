"""Source-level selection using frozen finite-completion diagnostics."""
import math

NAMES = ('mse', 'final', 'initial')


def choose(candidates):
    if set(candidates) != set(NAMES):
        raise ValueError('All three frozen source candidates required')
    eligible = []
    for name, row in candidates.items():
        bound, old = row['bound'], row['old']
        if bound['finite_completion_supported']:
            values = [bound[k] for k in ('all_selected_risk_upper', 'easy_selected_risk_upper',
                'easy_degradation_upper', 'selected_net_gain_lower_mass')]
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError('Finite supported bounds required')
            if (bound['reasons'] or bound['all_selected_risk_upper'] is None
                    or bound['easy_selected_risk_upper'] is None
                    or bound['easy_degradation_upper'] is None
                    or bound['selected_net_gain_lower_mass'] <= 0
                    or max(bound['all_selected_risk_upper'], bound['easy_selected_risk_upper'],
                           bound['easy_degradation_upper']) > .02+1e-12
                    or old['all_gain_fraction'] is None
                    or not math.isfinite(old['all_gain_fraction'])
                    or old['all_gain_fraction'] <= 0 or old['selected_count'] <= 0):
                raise ValueError('Inconsistent finite-completion support')
            eligible.append(name)
    selected = min(eligible, key=lambda name: (-candidates[name]['old']['all_gain_fraction'],
        candidates[name]['old']['selected_count'], NAMES.index(name))) if eligible else 'fallback'
    return dict(selected=selected, eligible=eligible,
        ranking='unchanged_known_validation_gain_not_guaranteed_relative_gain',
        independent_calibration=False, per_row_label_inference_gate=False)
