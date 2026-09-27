"""Cross fitting-row regime and train-derived easy-cut definitions."""
import numpy as np
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_nested_residual import labels, fitting_sites
from src.world_model.m3w_easy_membership_probe import labels as easy_labels

CELLS = ('two_cut2', 'two_cut3', 'three_cut2', 'three_cut3')
NEW_CELLS = ('two_cut3', 'three_cut2')


def inputs(x, raw, cv, sites, outer, omitted, cell):
    names = fitting_sites(sites, outer)
    if omitted not in names or cell not in CELLS:
        raise ValueError('Registered row/cut cell and omitted fitting locality required')
    x, raw, cv, sites = map(np.asarray, (x, raw, cv, sites))
    row_mask = sites != omitted if cell.startswith('two_') else np.ones(len(sites), bool)
    cut_mask = sites != omitted if cell.endswith('cut2') else np.ones(len(sites), bool)
    pr = preprocess(x[row_mask], raw[row_mask], cv[row_mask], sites[row_mask], outer)
    positive = cv[cut_mask & np.isfinite(cv) & (cv > 0)]
    if not len(positive):
        raise ValueError('Positive fitting baseline costs required for cut')
    cut = float(np.quantile(positive, .25))
    pr = dict(pr, positive_easy_cut=cut)
    y = labels(raw[row_mask], cv[row_mask], cut)
    easy = easy_labels(cv[row_mask], cut)
    lineage = dict(row_sites=sorted(set(sites[row_mask])),
        preprocessing_sites=pr['training_sites'], cut_sites=sorted(set(sites[cut_mask])),
        omitted=omitted, outer=outer, cell=cell,
        inference_feature_contains_cut=False, eligible_as_inner_OOF=cell == 'two_cut2')
    check_prediction_sites([outer], lineage)
    return row_mask, pr, y, easy, lineage


def check_prediction_sites(sites, lineage):
    producers = set().union(*(set(lineage[k]) for k in ('row_sites', 'preprocessing_sites', 'cut_sites')))
    if len(sites) == 0 or set(sites) & producers or lineage['outer'] in producers:
        raise ValueError('Prediction overlaps any parameter or label-definition producer')


def check_cut_match(a, b):
    """Changing the cut leaves draws, non-event scale and preprocessing fixed."""
    for key in ('draws', 'fixed_ids'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('mean', 'std', 'known', 'weights'):
        np.testing.assert_array_equal(a['preprocess'][key], b['preprocess'][key])
    for key in ('cost_scale', 'hard_cut', 'training_sites'):
        assert a['preprocess'][key] == b['preprocess'][key]
    np.testing.assert_array_equal(a['loss_scales'][:2], b['loss_scales'][:2])
    assert a['settings'] == b['settings'] and a['seed'] == b['seed'] and a['step'] == b['step']
    assert a['sampler_rng'].equal(b['sampler_rng'])


def mean_replica_deltas(values):
    if len(values) != 3 or set(values[0]) != set(values[1]) or set(values[0]) != set(values[2]):
        raise ValueError('All three fixed omitted-locality replicas required')
    return {k:float(np.mean([r[k] for r in values]))
        if all(r[k] is not None and np.isfinite(r[k]) for r in values) else None
        for k in values[0]}
