"""Exact no-op exclusion for a label-aware causal action-bank diagnostic."""
import numpy as np
from src.world_model.m3w_geometric_cost_head import rollout_envelope


def real_intervention_mask(candidate, floor):
    candidate, floor = np.asarray(candidate), np.asarray(floor)
    if (candidate.ndim != 3 or candidate.shape[-1] != 2 or candidate.shape != floor.shape
            or not np.isfinite(candidate).all() or not np.isfinite(floor).all()):
        raise ValueError('Aligned finite complete causal rollouts required')
    # Analytic baseline identities, not a fitted/tuned disagreement threshold.
    return np.any(candidate != floor, axis=(1, 2))


def causal_bank_envelope(cv, floor, candidate):
    # Partial future labels may select any step, so a mean is not an upper bound.
    return np.maximum(rollout_envelope(cv, candidate), rollout_envelope(floor, candidate))
