"""Factorial diagnostics for missing-row debits; not an evaluation rule."""
import numpy as np

from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_recording_switch_ceiling import BUDGET
from src.evaluation.m3w_selection_forecast_readout import errors, policy

ARMS = {
    'original': (True, True),
    'omit_utility_debit': (True, False),
    'omit_risk_debit': (False, True),
    'omit_both_debits': (False, False),
}


def variant_masses(base, cv, ols, bank, y, valid, weight, arm):
    risk_debit, utility_debit = ARMS[arm]
    m = dict(base, risk=base['risk'].copy(), utility=base['utility'].copy(),
             offset=base['offset'].copy())
    known = valid.any(1)
    unknown = ~known[m['row']]
    if not risk_debit:
        rows, cols = m['row'][unknown], m['arm'][unknown]
        radius = np.linalg.norm(bank[rows, cols]-cv[rows], axis=-1).max(1)
        m['risk'][:2, unknown] -= radius
    if not utility_debit:
        a, _ = errors(cv, y, valid)
        b, _ = errors(ols, y, valid)
        m['offset'][1] = float(weight[known] @ (b[known]-a[known]))
        m['utility'][:, unknown] = 0.
    return m


def evidence(cv, ols, pred, y, valid, rec, cut, weight, arm):
    risk_debit, utility_debit = ARMS[arm]
    risk = policy(pred, cv, cv, y, valid, np.full(len(cv), rec),
                  easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][rec]
    positive_keys = (('completion_all_risk_upper', 'completion_easy_risk_upper') if risk_debit
                     else ('known_all_risk', 'known_easy_risk'))
    checks = [risk[k] for k in (*positive_keys, 'known_easy_net_degradation')]
    supported = all(v is not None and v <= BUDGET for v in checks)
    if risk_debit and supported != risk['supported']:
        raise ValueError('Original recording risk verdict changed')
    values = [comparison_masses(p, pred, cv, y, valid, weight) for p in (cv, ols)]
    gains = [v['known_delta']-(v['unknown_radius'] if utility_debit else 0.) for v in values]
    return dict(supported=supported, gains=gains, denominator=values[0]['reference'])
