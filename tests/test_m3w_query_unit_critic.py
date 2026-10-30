import numpy as np
import torch
from src.world_model import m3w_query_unit_critic as api
from tests.test_m3w_selected_query_critic import fixture, settings


def data():
    c, y = fixture()
    return api.parent.training_data(c, y)


def test_query_rms_uses_only_complete_nonempty_original_training_selected_sets():
    d = data(); norm = api.query_normalizer(d)
    np.testing.assert_allclose(norm['rms'], [.03, .016])
    assert norm['sources_with_selected_support'] == 1
    assert norm['complete_selected_queries'] == 2
    assert norm['unknown_selected_queries'] == 1 and norm['empty_selected_queries'] == 1
    assert norm['fitting_labels_only'] and not norm['inference_feature']


def test_all_action_wrapper_reproduces_unmodified_real_training_exactly(tmp_path):
    d = data(); kw = dict(arm='selected_set', seed=17, settings=settings(), identity=dict(test=True))
    old = api.parent.fit(d, path=tmp_path/'old.pt.gz', **kw)
    control = api.fit(d, normalization='all_action', path=tmp_path/'control.pt.gz', **kw)
    for k in old:
        if k not in ('identity', 'seconds'): api.parent.exact(old[k], control[k])


def test_new_units_keep_point_loss_and_only_change_query_term():
    d = data(); c = d['causal']; ix = np.arange(8); seg = torch.repeat_interleave(torch.arange(4), 2)
    args = api.parent.batch(c, ix, seg, 4)
    pred = api.parent.initialize(4, 8, 17)(*args, arm='selected_set')
    old = api.ORIGINAL_LOSS(pred, d, ix, seg, 4)
    with api.installed(api.query_normalizer(d), 'selected_query'):
        new = api.parent.losses(pred, d, ix, seg, 4)
    torch.testing.assert_close(new['pointwise_risk'], old['pointwise_risk'], rtol=0, atol=0)
    torch.testing.assert_close(new['selected_query_score_original_units'], old['selected_query_score'], rtol=0, atol=0)
    assert new['selected_query_score'] > old['selected_query_score']
    assert api.parent.losses is api.ORIGINAL_LOSS


def test_actual_query_unit_fit_recovers_exactly_and_keeps_paired_design(tmp_path):
    d = data()
    def fit(arm, path, **kw):
        return api.fit(d, normalization='selected_query', arm=arm, seed=17,
            settings=settings(), identity=dict(test=True), path=path, **kw)
    direct = fit('selected_set', tmp_path/'direct.pt.gz')
    fit('selected_set', tmp_path/'resume.pt.gz', stop_at=2)
    resumed = fit('selected_set', tmp_path/'resume.pt.gz', resume=True)
    for k in direct:
        if k != 'seconds': api.parent.exact(direct[k], resumed[k])
    local = fit('row_local', tmp_path/'local.pt.gz')
    api.parent.assert_matched(direct, local)
    assert direct['unknown_rows_with_loss'] == 0
    assert 'selected_query_score_original_units' in direct['trace'][-1]['monitor']
