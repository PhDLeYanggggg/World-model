import inspect
import json
import numpy as np
from scripts import run_m3w_european_fixed_floor_slices as run


def test_registered_scope_does_not_train_select_or_open_independent_roles():
    cfg = json.loads((run.ROOT/run.CONFIG).read_text())
    assert cfg['groups'] == 108 and cfg['bootstrap_resamples'] >= 2000
    assert cfg['risk_budget'] == .02 and cfg['causal_quantiles'] == [.25, .75]
    assert not any(cfg[k] for k in ('new_training', 'threshold_search', 'independent_roles_read',
        'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    assert not any(k in inspect.signature(run.api.causal_axes).parameters for k in ('future', 'valid', 'target'))


def test_evaluation_strata_never_alter_causal_bins():
    axes = dict(feature_radius_over_limit=np.array([.5, 1., 2.]))
    edges = dict(feature_radius_over_limit=[.5, 1.5])
    valid = np.ones((3, 12), bool); valid[0, 3:] = False; valid[2] = False
    before = run.sliced_masks(axes, edges, np.array([1., 2., np.nan]), [1, 2], valid, 1.)
    after = run.sliced_masks(axes, edges, np.array([3., np.nan, 1.]), [1, 2], valid[[0, 2, 1]], 1.)
    for a, b in zip(before, after):
        if 'eval_only' not in a[0]: np.testing.assert_array_equal(a[2], b[2])
    labels = {b: m for a, b, m in before if a == 'label_completeness_eval_only'}
    assert labels['partial'].tolist() == [True, False, False]
    assert labels['complete'].tolist() == [False, True, False]
    assert labels['unknown'].tolist() == [False, False, True]
