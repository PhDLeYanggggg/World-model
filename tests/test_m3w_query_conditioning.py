import copy

import numpy as np
import pytest

from src.evaluation import m3w_query_conditioning as audit
from src.evaluation.m3w_selected_tail_audit import actions
from tests.test_m3w_query_context_risk import anchor, fit
from tests.test_m3w_decoder_risk_factorial import packets


def test_query_autograd_matches_inference_and_preserves_state(tmp_path):
    a = anchor(tmp_path); state = fit(tmp_path/'peer.pt.gz', a, 'peer')
    old = copy.deepcopy(state); args = packets()['a']
    values, _ = audit.collect(state, 'a', args[0], args[1], args[5], args[6])
    expected = audit.api.predict(state, 'a', args[0], args[1], args[5], args[6])
    np.testing.assert_array_equal(values['prediction'], expected)
    assert values['jacobian'].shape == (len(args[0]), 2, 2)
    assert (values['jacobian'][:, 0, 1] == 0).all()
    audit.api.core.exact(old, state)


def test_unknown_rows_and_signed_residuals_are_explicit():
    y = np.array([[0, .2, 1, 1, .2], [np.nan]*5, [0, 0, 1, 1, 0]])
    p = np.tile([.1, .01, 1., 1., .005], (3, 1))
    values = dict(prediction=p, jacobian=np.zeros((3, 2, 2)), cap=np.ones(3),
                  anchor_h=np.full(3, .1), anchor_q=np.full(3, .1), raw=np.zeros((3, 2)))
    out = audit.account(values, y, np.ones(3, bool), 1.)
    assert out['known'] == 2 and out['unknown'] == 1
    assert out['easy']['false_safe_rows'] == 1
    assert out['easy']['residual_sum'] == pytest.approx(.19)
    assert out['easy']['positive_residual'] == pytest.approx(.195)


def test_virtual_steps_keep_checkpoint_and_optimizer_unchanged(tmp_path):
    a = anchor(tmp_path); state = fit(tmp_path/'self.pt.gz', a, 'self')
    original = copy.deepcopy(state); batch = audit.prepare_probes(packets(), state, count=1)[0]
    common = {}
    for s, d in batch.items():
        args = packets()[s]
        p = audit.api.predict(state, s, args[0], args[1], args[5], args[6])
        common[s] = audit.torch.from_numpy(actions(p, audit.api.parent.previous.prior.eligible(args))[d['row_ids']])
    r = audit.gradient_probe(state, batch, common)
    assert set(r['gradient_norms']) == {'primary', 'regret', 'dual_all', 'dual_easy'}
    assert r['virtual_steps']['as_trained']['optimizer_step'] == state['step']+1
    assert 0 < r['clip_factor'] <= 1
    audit.api.core.exact(state, original)
    # Reconstructing probes starts from the same copied final sampler state.
    repeated = audit.prepare_probes(packets(), state, count=1)[0]
    for s in batch:
        np.testing.assert_array_equal(batch[s]['row_ids'], repeated[s]['row_ids'])


def test_incomplete_rosters_rejected():
    with pytest.raises(ValueError):
        audit.summarize([], [])
