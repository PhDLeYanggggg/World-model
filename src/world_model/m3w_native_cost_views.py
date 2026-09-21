"""Materialize explicit source head-training partitions, never deployment gates."""
from __future__ import annotations

import numpy as np

from src.world_model.m3w_native_nested import require_source_producer


LABEL_KEYS = ('neural_ade', 'baseline_ade', 'neural_fde', 'baseline_fde',
              'gain', 'benefit', 'harm', 'supported_steps', 'complete_future')


def assemble_training_view(outer_site, seed, sites, groups):
    sites = np.asarray(sites)
    roster = set(sites)
    if outer_site not in roster or len(groups) != len(roster)-1:
        raise ValueError('Complete inner roster required')
    inners = [g['inner_site'] for g in groups]
    if set(inners) != roster-{outer_site} or len(set(inners)) != len(inners):
        raise ValueError('Duplicate, missing or outer training group')
    all_ids, all_p = [], []
    collected = {k:[] for k in LABEL_KEYS}
    for g in groups:
        inner = g['inner_site']
        require_source_producer(g['producer'], outer_site=outer_site, row_site=inner, roster=roster, seed=seed)
        ids, prediction, labels = np.asarray(g['ids']), np.asarray(g['prediction']), g['labels']
        if (ids.dtype.kind not in 'iu' or ids.ndim != 1
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(sites))
                or prediction.shape != (len(ids), 12, 2) or set(labels) != set(LABEL_KEYS)
                or any(np.shape(labels[k]) != ids.shape for k in LABEL_KEYS)):
            raise ValueError('Invalid prediction/label archive schema')
        expected = np.flatnonzero(np.isin(sites, [inner, outer_site]))
        np.testing.assert_array_equal(ids, expected)
        keep = sites[ids] == inner
        use = ids[keep]
        np.testing.assert_array_equal(use, np.flatnonzero(sites == inner))
        if not np.isfinite(prediction[keep]).all():
            raise ValueError('Nonfinite training prediction')
        all_ids.append(use); all_p.append(prediction[keep])
        selected = {k:np.asarray(labels[k])[keep] for k in LABEL_KEYS}
        steps = selected['supported_steps']
        if (steps.dtype.kind not in 'iu' or np.any((steps < 0) | (steps > 12))
                or selected['complete_future'].dtype != bool
                or not np.array_equal(selected['complete_future'], steps == 12)):
            raise ValueError('Invalid supervision support metadata')
        support = selected['supported_steps'] > 0
        for k in ('neural_ade', 'baseline_ade', 'gain', 'benefit', 'harm'):
            if np.isinf(selected[k]).any() or not np.array_equal(np.isnan(selected[k]), ~support):
                raise ValueError('Unknown ADE supervision must remain unknown')
        np.testing.assert_allclose(selected['baseline_ade']-selected['neural_ade'], selected['gain'], equal_nan=True)
        np.testing.assert_allclose(selected['benefit']-selected['harm'], selected['gain'], equal_nan=True)
        if any(np.any(selected[k][support] < 0) for k in ('baseline_ade', 'neural_ade', 'benefit', 'harm')):
            raise ValueError('Negative cost')
        for k in LABEL_KEYS:
            collected[k].append(selected[k])
    ids = np.concatenate(all_ids); order = np.argsort(ids)
    ids = ids[order]
    np.testing.assert_array_equal(ids, np.flatnonzero(sites != outer_site))
    # Row IDs are alignment metadata. Only the causal prediction is model input.
    inputs = dict(ids=ids, prediction=np.concatenate(all_p)[order])
    targets = dict(ids=ids, **{k:np.concatenate(v)[order] for k,v in collected.items()})
    return inputs, targets
