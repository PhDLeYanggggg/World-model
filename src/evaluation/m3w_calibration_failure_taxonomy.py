"""Descriptive calibration failure causes; never an inference or fitting rule."""
import math

CATEGORIES = ('unsupported_reference', 'observed_positive_harm', 'easy_net_only',
              'unknown_completion_bound_only', 'supported')


def classify(view):
    names = ('known_all_risk', 'known_easy_risk', 'completion_all_risk_upper',
             'completion_easy_risk_upper', 'known_easy_net_degradation')
    for key in names:
        value = view[key]
        if value is not None and not math.isfinite(value):
            raise ValueError('Unavailable ratios must be None, not nonfinite numeric values')
    unsupported = any(view[k] is None for k in names)
    harm = any(view[k] is not None and view[k] > .02 for k in names[:2])
    net = view[names[-1]] is not None and view[names[-1]] > .02
    completion = any(view[k] is not None and view[k] > .02 for k in names[2:4])
    expected = not (unsupported or harm or net or completion)
    if bool(view['supported']) != expected:
        raise ValueError('Original supported flag and component risks disagree')
    if unsupported:
        category = 'unsupported_reference'
    elif harm:
        category = 'observed_positive_harm'
    elif net:
        category = 'easy_net_only'
    elif completion:
        category = 'unknown_completion_bound_only'
    else:
        category = 'supported'
    return dict(category=category, unsupported_reference=unsupported,
        full_roster_easy_reference_unavailable=view['known_easy_net_degradation'] is None,
        selected_all_reference_unavailable=view['known_all_risk'] is None,
        selected_easy_reference_unavailable=view['known_easy_risk'] is None,
        observed_positive_harm_violation=harm, observed_easy_net_violation=net,
        completion_upper_violation=completion, selected_none=view['selected'] == 0,
        selected_unknown_outcome=view['unknown_selected'] > 0)
