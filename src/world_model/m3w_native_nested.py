"""Pair-excluded native producers and explicit cost-label provenance."""
from __future__ import annotations

import numpy as np

from src.evaluation.m3w_native_metrics import native_errors
from src.world_model.m3w_native_forecast import fold_design


def nested_fold_design(data, excluded_sites):
    sites = np.asarray(data['sites'])
    pair = tuple(sorted(excluded_sites))
    roster = set(sites)
    if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= roster or len(roster) < 4:
        raise ValueError('Two distinct excluded physical sites and at least two training sites required')
    keep = np.flatnonzero(sites != pair[0])
    subset = {k:np.asarray(data[k])[keep] for k in ('sites', 'geometry', 'target', 'valid', 'scale')}
    inner = fold_design(subset, pair[1], 'native_coordinate')
    factors = np.zeros(len(sites), np.float64)
    factors[keep] = inner['factors']
    train = keep[inner['train_ids']]
    held = np.flatnonzero(np.isin(sites, pair))
    np.testing.assert_array_equal(train, np.flatnonzero(~np.isin(sites, pair)))
    return dict(train_ids=train, held_ids=held,
        groups=[keep[g] for g in inner['groups']], factors=factors,
        normalizers=inner['normalizers'], hard_cut=inner['hard_cut'],
        objective='native_coordinate', excluded_sites=list(pair))


def require_source_producer(record, *, outer_site, row_site, roster, seed):
    """Check this fixed, random-init source design, not arbitrary ancestor graphs."""
    roster = set(roster)
    excluded = {outer_site, row_site}
    training = roster-excluded
    if (not excluded <= roster or record['seed'] != seed
            or set(record['excluded_sites']) != excluded
            or set(record['training_sites']) != training
            or set(record['preprocessing_fit_sites']) != training
            or record['parents'] or record['initialization'] != 'random_seed'
            or record['checkpoint_selection_sites'] or record['calibration_sites']
            or record['objective'] != 'native_coordinate'
            or set(record['research_design_exposed_sites']) != roster):
        raise ValueError('Producer violates fixed source fitting/preprocessing exclusion')
    return dict(outer_site=outer_site, row_site=row_site, producer=record['id'],
        seed=seed, fitting_exclusion_pass=True, independent_confirmation=False,
        research_design_exposed_sites=sorted(roster))


def cost_supervision(prediction, baseline, target, valid, scale):
    """Targets only: never merge these arrays into a predictor feature payload."""
    model, model_fde = native_errors(prediction, target, valid, scale)
    reference, reference_fde = native_errors(baseline, target, valid, scale)
    gain = reference-model
    return dict(neural_ade=model, baseline_ade=reference,
        neural_fde=model_fde, baseline_fde=reference_fde,
        gain=gain, benefit=np.maximum(gain, 0), harm=np.maximum(-gain, 0),
        supported_steps=np.asarray(valid).sum(1), complete_future=np.asarray(valid).all(1))
