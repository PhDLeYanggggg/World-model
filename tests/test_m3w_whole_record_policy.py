import copy

import numpy as np
import pytest
import torch

from src.world_model import m3w_whole_record_policy as api
from tests.test_m3w_direct_policy import fixture, settings


def data():
    return api.base.prepare(*fixture())


def test_chunked_full_gradient_matches_dense_objective():
    d = data(); layout = api.layout(d)
    model = api.base.Policy(d['z'].shape[1], 8, 17)
    dual = torch.full((len(d['groups']), 2), .7)
    dense = api.full_pass(model, d, layout, dual, .5, 1000, backward=True)
    first = torch.cat([p.grad.flatten() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    split = api.full_pass(model, d, layout, dual, .5, 2, backward=True)
    second = torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(dense['risk'], split['risk'], rtol=1e-6, atol=1e-8)
    ix, record = layout['index'], layout['record']
    action = api.base.decision(model(d['z'][ix], d['anchor'][ix]), True, .5)
    coeff = -d['gain'][ix]+(dual[record]*d['risk'][ix]/d['record_scales'][record]).sum(1)
    expected = (action*coeff*layout['weight']).sum()
    assert dense['loss'] == pytest.approx(float(expected.detach()), abs=1e-6)


def test_rare_harm_sample_can_look_safe_when_full_record_is_not():
    terms = torch.full((1000, 2), -.01); terms[0] = 100.
    out = api.sampling_diagnostic(terms, 64, 512, 17)
    assert min(out['exact_mean']) > 0
    assert min(out['miss_positive_probability']) > .9
    assert min(out['nonpositive_sample_fraction']) > .8
    assert out['positive_count'] == [1, 1]


@pytest.mark.parametrize('arm', api.ARMS)
def test_resume_preserves_exact_training_and_fixed_inference(tmp_path, arm):
    d = data(); cfg = {**settings(), 'chunk_size': 2}
    kw = dict(arm=arm, settings=cfg, seed=17, identity={'test': True})
    a = api.fit(d, path=tmp_path/'full.pt.gz', **kw)
    api.fit(d, path=tmp_path/'resume.pt.gz', stop_at=3, **kw)
    b = api.fit(d, path=tmp_path/'resume.pt.gz', resume=True, **kw)
    for key in a:
        if key != 'seconds': api.base.core.exact(a[key], b[key])
    assert any(not torch.equal(a['model'][k], v) for k,v in a['initial_model'].items())
    assert a['whole_record_forward_rows'] == int(sum(d['sizes']))*6
    assert a['optimizer_steps'] == 6 and not a['new_forecaster']


def test_exact_dual_keeps_original_primal_draws(tmp_path):
    d = data(); cfg = settings()
    original = api.base.fit(d, arm='hard_record', settings=cfg, seed=17,
                           identity={'test': True}, path=tmp_path/'old.pt.gz')
    new = api.fit(d, arm='exact_dual', settings={**cfg, 'chunk_size': 2}, seed=17,
                  identity={'test': True}, path=tmp_path/'new.pt.gz')
    assert new['draw_hash'] == original['draw_hash']
    assert new['row_draws'] == original['row_draws']
    api.base.core.exact(original['initial_model'], new['initial_model'])


def test_layout_rejects_duplicated_training_rows():
    d = copy.deepcopy(data()); d['groups'][1] = d['groups'][0]
    with pytest.raises(ValueError, match='disjoint'):
        api.layout(d)


def test_resume_between_heartbeat_boundaries_is_exact(tmp_path):
    d = data(); cfg = {**settings(), 'chunk_size': 2, 'heartbeat_every': 50}
    kw = dict(arm='exact_gradient', settings=cfg, seed=17, identity={'test': True})
    full = api.fit(d, path=tmp_path/'full.pt.gz', **kw)
    api.fit(d, path=tmp_path/'part.pt.gz', stop_at=2, **kw)
    resumed = api.fit(d, path=tmp_path/'part.pt.gz', resume=True, **kw)
    for key in full:
        if key != 'seconds': api.base.core.exact(full[key], resumed[key])
