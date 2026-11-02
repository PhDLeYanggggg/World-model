"""Future-informed necessary support diagnosis, never an inference policy."""
import numpy as np


def diagnose(target, allowed, recordings, budget=.02):
    y,legal,records = map(np.asarray,(target,allowed,recordings))
    if y.ndim != 3 or y.shape[1:] != (7,5) or legal.shape != y.shape[:2] or legal.dtype != bool or records.shape != (len(y),):
        raise ValueError('Aligned native seven-action cost labels and causal availability required')
    known = np.isfinite(y).all((1,2)); wholly_unknown = np.isnan(y).all((1,2))
    if not (known|wholly_unknown).all() or (y[known] < 0).any() or budget != .02:
        raise ValueError('Native known/wholly-unknown costs and unchanged2%contract required')
    for k in (2,3):
        np.testing.assert_array_equal(y[known,:,k],np.repeat(y[known,0,k,None],7,axis=1))
    if ((y[known,:,3] != 0)&(y[known,:,3] != y[known,:,2])).any():
        raise ValueError('Native easy-reference event must be zero or the same reference')
    eligible = legal & known[:,None] & (y[:,:,3] > 0)
    safe = eligible & (y[:,:,1] <= budget*y[:,:,2]) & (y[:,:,4] <= budget*y[:,:,3])
    output = {}
    for rec in sorted(set(records)):
        at = records == rec; mass = float(y[at&known,0,3].sum())
        candidates = int(eligible[at].sum()); passing = int(safe[at].sum())
        reason = ('one_known_easy_action_suffices_without_additional_count_constraints' if passing else
            'no_positive_known_easy_reference' if mass <= 0 else
            'no_genuine_legal_easy_action' if not candidates else
            'all_legal_known_easy_actions_exceed_original_budget')
        output[str(rec)] = dict(rows=int(at.sum()),known_rows=int((at&known).sum()),
            wholly_unknown_rows=int((at&~known).sum()),full_known_easy_reference_mass=mass,
            legal_known_easy_action_occurrences=candidates,safe_known_easy_action_occurrences=passing,
            strictly_beneficial_safe_occurrences=int((safe[at]&(y[at,:,0] > 0)).sum()),
            finite_support_bank_feasible=bool(passing),reason=reason)
    return dict(recordings=output,required_record_views=len(output),
        bank_infeasible_record_views=sum(not v['finite_support_bank_feasible'] for v in output.values()),
        future_informed_diagnostic_not_causal_model=True,oracle_decisions_exported=False,
        original_policy_risk_reclassified=False,partial_mask_cost_not_full12_error=True)
