import numpy as np
import pytest
import torch
from src.world_model import m3w_current_policy_critic as api
from tests.test_m3w_query_unit_critic import data
from tests.test_m3w_selected_query_critic import settings


def test_current_selection_uses_prediction_and_causal_fields_not_future_labels():
    d = data(); c = d['causal']; rows = np.arange(8); seg = torch.repeat_interleave(torch.arange(4), 2)
    p = api.unit.parent.initialize(4, 8, 17)(*api.unit.parent.batch(c, rows, seg, 4), arm='selected_set')
    choice, info = api.selected_actions(p, c, rows, seg, 4, seconds=2.)
    assert len(choice) == 8 and info['query_count'] == 4
    assert choice[1] >= 0 and not d['known'][1]
    assert info['labels_used'] is False
    with pytest.raises(ValueError): api.selected_actions(p, dict(c, targets=d['targets']), rows, seg, 4, seconds=2.)


def test_fixed_target_wrapper_reproduces_query_unit_control_exactly(tmp_path):
    d = data(); kw = dict(arm='selected_set', seed=17, settings=settings(), identity=dict(test=True))
    old = api.unit.fit(d, normalization='selected_query', path=tmp_path/'old.pt.gz', **kw)
    fixed = api.fit(d, target='frozen_original', path=tmp_path/'fixed.pt.gz', **kw)
    for key in old:
        if key not in ('identity', 'seconds'): api.unit.parent.exact(old[key], fixed[key])


def test_only_query_target_changes_and_unknowns_never_supervise():
    d = data(); c = d['causal']; rows = np.arange(8); seg = torch.repeat_interleave(torch.arange(4), 2)
    p = api.unit.parent.initialize(4, 8, 17)(*api.unit.parent.batch(c, rows, seg, 4), arm='selected_set')
    norm = api.unit.query_normalizer(d)
    with api.unit.installed(norm, 'selected_query'): old = api.unit.parent.losses(p, d, rows, seg, 4)
    with api.installed(norm, 'current_decoded', seconds=2.): new = api.unit.parent.losses(p, d, rows, seg, 4)
    torch.testing.assert_close(old['pointwise_risk'], new['pointwise_risk'], rtol=0, atol=0)
    assert new['unknown_supervised_rows'] == 0
    assert new['current_selected_queries_with_unknown_labels'] == 1
    assert new['current_selected_rows'] >= 1 and new['selected_actions_detached']
    assert 'frozen_selected_query_score_original_units' in new
    assert api.unit.parent.losses is api.unit.ORIGINAL_LOSS


def test_actual_current_policy_training_resumes_exactly_and_keeps_matched_draws(tmp_path):
    d = data(); kw = dict(seed=17, settings=settings(), identity=dict(test=True), target='current_decoded')
    def fit(arm, path, **more): return api.fit(d, arm=arm, path=path, **kw, **more)
    direct = fit('selected_set', tmp_path/'direct.pt.gz')
    fit('selected_set', tmp_path/'resumed.pt.gz', stop_at=2)
    recovered = fit('selected_set', tmp_path/'resumed.pt.gz', resume=True)
    for key in direct:
        if key != 'seconds': api.unit.parent.exact(direct[key], recovered[key])
    local = fit('row_local', tmp_path/'row.pt.gz')
    api.unit.parent.assert_matched(direct, local)
    assert recovered['unknown_rows_with_loss'] == 0


def test_incomplete_training_optimization_refuses_instead_of_silent_nondeterminism(monkeypatch):
    d = data(); c = d['causal']; rows = np.arange(8); seg = torch.repeat_interleave(torch.arange(4), 2)
    p = api.unit.parent.initialize(4, 8, 17)(*api.unit.parent.batch(c, rows, seg, 4), arm='selected_set')
    monkeypatch.setattr(api.decoder.base, 'solve_frame', lambda q, ok, anchor, **kw:
        (anchor, dict(reason='solver_status_1_parent_fallback', optimal=False)))
    with pytest.raises(RuntimeError): api.selected_actions(p, c, rows, seg, 4, seconds=2.)
