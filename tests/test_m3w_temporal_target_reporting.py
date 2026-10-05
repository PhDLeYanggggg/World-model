import numpy as np

from scripts import run_m3w_temporal_target_audit as runner
from scripts.verify_m3w_temporal_target_audit import aggregate


def test_zero_reference_easy_harm_is_not_dropped():
    ref = np.zeros((1, 12, 2)); candidate = ref.copy(); candidate[..., 0] = 1
    _, d = runner.api.targets(ref, candidate, ref, np.ones((1, 12), bool))
    y = np.array([[0., 1., 0., 0., 1.]])
    out = runner.decomposition(d, y, np.array([True]))
    assert out['known_selected_easy_harm'] == 1.
    assert out['known_selected_easy_step_harm'] == 1.
    assert out['original_known_easy_ratio'] is None


def test_identical_probes_fail_screen_and_match_independent_intervals():
    cfg = dict(bootstrap_resamples=3000, bootstrap_seed=20261005)
    fields = ['rows', 'unknown_rows', 'full_label_rows', 'selected', 'unknown_selected',
              'selected_opposite_sign', 'selected_netharm_rows', 'selected_netharm_with_cancellation',
              'selected_nonharmful_but_step_harmful', 'selected_harm', 'selected_step_harm', 'selected_cancellation']
    rows = []
    for site in ('a', 'b', 'c'):
        m = {k: [1., 2.] for k in ('temporal_leaf', 'rowmean_leaf', 'global_temporal')}
        rows.append(dict(source=site, probes={k: m for k in ('validation', 'complete_validation', 'selected_validation')},
                         decomposition=dict(validation={k: 0 for k in fields})))
    summary = runner.summary(rows, cfg)
    assert not summary['advance_to_auxiliary_design']
    assert len(summary['failure_reasons']) == 8
    assert summary['contrasts'] == aggregate(rows, cfg)
