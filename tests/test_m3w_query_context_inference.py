import numpy as np

from src.world_model import m3w_query_context_risk as api
from tests.test_m3w_isolated_harm_fp64 import anchor
from tests.test_m3w_query_context_risk import fit
from tests.test_m3w_decoder_risk_factorial import packets


def test_pooling_happens_before_inference_batches(tmp_path):
    a = anchor(tmp_path)
    state = fit(tmp_path/'peer.pt.gz', a, 'peer')
    args = packets()['a']
    x, env, recs = args[0], args[1], args[5]
    frames = np.zeros(len(x))
    full = api.predict(state, 'a', x, env, recs, frames)
    chunked = api.predict(state, 'a', x, env, recs, frames, batch_size=3)
    np.testing.assert_allclose(chunked, full, rtol=2e-5, atol=1e-7)
    order = np.random.default_rng(74).permutation(len(x))
    shuffled = api.predict(state, 'a', x[order], env[order], recs[order], frames[order])
    np.testing.assert_allclose(shuffled, full[order], rtol=2e-5, atol=1e-7)


def test_foreign_recordings_cannot_change_existing_query(tmp_path):
    a = anchor(tmp_path)
    state = fit(tmp_path/'peer.pt.gz', a, 'peer')
    args = packets()['a']; x, env, recs = args[0], args[1], args[5]
    frames = np.zeros(len(x))
    full = api.predict(state, 'a', x, env, recs, frames)
    changed = x.copy(); changed[recs != '0'] *= -100
    other = api.predict(state, 'a', changed, env, recs, frames)
    np.testing.assert_array_equal(other[recs == '0'], full[recs == '0'])
