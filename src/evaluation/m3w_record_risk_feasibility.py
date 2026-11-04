"""Label-aware feasibility of the fixed binary proposal, not a causal policy."""
import numpy as np


def witness(reference, candidate, cv, allowed, recordings, *, easy_cut, budget=.02):
    reference,candidate,cv = map(lambda v:np.asarray(v,float),(reference,candidate,cv))
    allowed,recordings = np.asarray(allowed),np.asarray(recordings)
    n = len(reference)
    if (reference.shape != (n,) or candidate.shape != (n,) or cv.shape != (n,)
            or allowed.shape != (n,) or allowed.dtype != bool or recordings.shape != (n,)
            or not n or budget != .02 or not np.isfinite(easy_cut) or easy_cut <= 0):
        raise ValueError('Fixed aligned costs, causal eligibility and original2%budget required')
    known = np.isfinite(reference)
    if (not np.array_equal(known,np.isfinite(candidate)) or not np.array_equal(known,np.isfinite(cv))
            or np.isinf(reference).any() or np.isinf(candidate).any() or np.isinf(cv).any()
            or any((v[known] < 0).any() for v in (reference,candidate,cv))):
        raise ValueError('Costs need identical known/wholly-unknown support; unknown is never zero')
    easy = known & (cv > 0) & (cv <= easy_cut)
    harm = np.maximum(candidate-reference,0)
    ratio = np.divide(harm,reference,out=np.full(n,np.inf),where=known & (reference > 0))
    choices = np.zeros(n,bool)
    records = {}
    for recording in sorted(set(recordings)):
        take = recordings == recording
        positive = take & easy & (reference > 0)
        legal = positive & allowed
        safe = legal & (ratio <= budget)
        best = int(np.flatnonzero(safe)[np.argmin(ratio[safe])]) if safe.any() else None
        full_easy_reference = float(reference[take & easy].sum())
        reason = ('feasible_label_aware_witness' if best is not None else
            'no_known_positive_easy_reference' if not positive.any() else
            'no_legal_positive_easy_reference' if not legal.any() else
            'every_legal_easy_harm_ratio_exceeds_original_budget')
        if best is not None:
            choices[best] = True
        records[str(recording)] = dict(rows=int(take.sum()),known_rows=int((take & known).sum()),
            unknown_rows=int((take & ~known).sum()),known_easy_rows=int((take & easy).sum()),
            positive_easy_reference_rows=int(positive.sum()),legal_positive_easy_reference_rows=int(legal.sum()),
            safe_easy_witness_rows=int(safe.sum()),full_known_easy_reference_mass=full_easy_reference,
            minimum_legal_easy_harm_ratio=float(ratio[legal].min()) if legal.any() else None,
            binary_fixed_proposal_finite_risk_feasible=best is not None,reason=reason)
    return choices,records
