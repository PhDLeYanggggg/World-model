"""Fixed-policy, post-hoc cost diagnostics without fitting or future inputs."""
import numpy as np

from src.evaluation.m3w_opportunity_diagnosis import causal_reasons, opportunity_ledger
from src.world_model.m3w_european_conditional_risk import event_labels
from src.world_model.m3w_european_source_intervention import paired_cost_labels


def validate_roles(head_sites, producer_sites, readout_sites):
    h, p, r = map(set, (head_sites, producer_sites, readout_sites))
    if len(h) != 4 or len(p) not in (2, 4) or len(r) != 8 or not p <= h or h & r:
        raise ValueError('Four fitting sites, nested producer and eight excluded sources required')


def score_diagnosis(reference, candidate, costs, moments, moving, sites, *, easy_cut,
                    event, budget=.02, resamples=3000, seed=39271):
    reference, candidate, costs, moments, sites = map(np.asarray,
        (reference, candidate, costs, moments, sites))
    labels = paired_cost_labels(reference, candidate)
    truth = event_labels(reference, labels[:, 1], easy_cut=easy_cut, event=event)
    if costs.shape != labels.shape or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError('Aligned finite nonnegative causal gain/harm scores required')
    why = causal_reasons(costs[:, 0]-costs[:, 1], moments, moving,
        np.ones(len(reference), bool), budget=budget)
    ledger = opportunity_ledger(reference, candidate, why, sites, expected_scenes=sorted(set(sites)),
        resamples=resamples, seed=seed)
    rows = {}
    for site in sorted(set(sites)):
        use = (sites == site) & np.isfinite(reference)
        selected = use & (why == 5)
        slices = {}
        for name, mask in [('population', use), ('selected', selected)]:
            if not mask.any():
                slices[name] = dict(rows=0, mean_cv=None, utility=None, risk=None)
                continue
            scale = float(reference[mask].mean())
            fields = {}
            for kind, predicted, target in [('utility', costs, labels), ('risk', moments, truth)]:
                error = predicted[mask]-target[mask]
                fields[kind] = dict(predicted_mean=predicted[mask].mean(0).tolist(),
                    actual_mean=target[mask].mean(0).tolist(), bias=error.mean(0).tolist(),
                    mae=np.abs(error).mean(0).tolist(),
                    mae_over_mean_cv=(np.abs(error).mean(0)/scale).tolist() if scale > 0 else None)
            slices[name] = dict(rows=int(mask.sum()), mean_cv=scale, **fields)
        rows[site] = slices
    return dict(ledger=ledger, score_errors=rows, switch_rate=float((why == 5).mean()),
        reasons=why, diagnostic_only=True, future_labels_used_for_scoring_only=True)
