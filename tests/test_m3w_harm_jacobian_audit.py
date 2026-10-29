import copy
import numpy as np
import pytest

from src.evaluation import m3w_harm_jacobian_audit as audit
from src.world_model import m3w_full_range_harm as api
from tests.test_m3w_full_range_harm import TinyBase
from tests.test_m3w_isolated_harm_fp64 import anchor
from tests.test_m3w_shared_train_risk import packet


def test_hard_clipping_kills_actual_derivative():
    model = api.HarmBranch(TinyBase(), 2, True)
    capture = {}
    hook = model.risk_output.register_forward_hook(lambda _m, _i, o: capture.update(raw=o))
    with api.torch.no_grad():
        model.risk_output.bias.fill_(-.5)
    out = model(api.torch.zeros(3, 2), api.torch.ones(3))
    j = api.torch.autograd.grad(out[:, 4].sum(), capture['raw'])[0]
    assert (out[:, [1, 4]] == 0).all() and (j == 0).all()
    hook.remove()


def test_collect_replays_predictions_without_modifying_checkpoint(tmp_path):
    a = anchor(tmp_path)
    m = api.initialize(a, 'isolated_risk')
    with api.torch.no_grad():
        m.risk_output.bias.copy_(api.torch.tensor([.2, -.2]))
    state = dict(anchor_state=a, model=m.state_dict(), arm='isolated_risk')
    before = copy.deepcopy(state)
    x, env = packet()[:2]
    v = audit.collect(api, state, 'a', x, env)
    np.testing.assert_array_equal(v['prediction'], api.predict(state, 'a', x, env))
    api.core.exact(state, before)
    assert v['jacobian'].shape == (len(x), 2, 2)


def test_unknown_retained_and_error_attribution():
    v = dict(prediction=np.zeros((3, 5)), jacobian=np.zeros((3, 2, 2)),
        cap=np.ones(3), anchor_h=np.full(3, .1), anchor_q=np.full(3, .1), raw=np.full((3, 2), -.5))
    y = np.ones((3, 5)); y[1] = np.nan
    out = audit.account(v, y, np.ones(3, bool), 1., 'full_range')
    assert out['unknown'] == 1 and out['known'] == 2
    assert out['easy']['underprediction_zero_jacobian'] == 2
    assert out['causes']['hard_h_lower'] == 3
    assert out['causes']['hard_q_lower'] == 3
    out['known'] = 3
    with pytest.raises(ValueError):
        audit.validate(out)


def test_summarize_rejects_incomplete_or_held():
    with pytest.raises(ValueError):
        audit.summarize([])
    with pytest.raises(ValueError):
        audit.summarize([dict(context=str(i), source='s', seed=1, scope='held') for i in range(72)])
