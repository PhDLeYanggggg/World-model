import numpy as np
import pytest
import torch

from src.world_model.m3w_visual_forecast import VisualSourceForecaster, normalizer, pack_inputs
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import fit_trial
from src.training.m3w_visual_forecast import fit


ARCH = dict(family='eqmotion_fixed_head', history_steps=8, prediction_steps=12,
    hidden_nf=8, channels=8, layers=2, fixed_head=0, prune_unused_heads=True,
    input_conditioning='observed_joint_max_norm')


def geometry(n=4):
    rng = np.random.default_rng(17); g = np.zeros((n, 476), np.float32)
    g[:, :16] = rng.normal(size=(n, 16)); g[:, 14:16] = 0
    g[:, 16:24] = np.arange(-7, 1)/12
    g[:, 38:166] = rng.normal(size=(n, 128))
    g[:, 166:230] = np.tile(np.arange(-7, 1)/12, 8)
    g[:, 230:294] = 1
    return g


def test_actual_author_core_zero_context_branch_matches_old_function_and_rng():
    from src.world_model.m3w_native_forecast import pack_geometry
    torch.manual_seed(17); old = SourceForecaster(ARCH, 1); state = torch.get_rng_state().clone()
    torch.manual_seed(17); new = VisualSourceForecaster(ARCH, 1, np.zeros(46), np.ones(46), 'motion_geometry')
    assert torch.equal(state, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    g = geometry(); x = np.random.default_rng(2).normal(size=(4, 46)).astype(np.float32)
    old.eval(); new.eval()
    with torch.no_grad():
        torch.testing.assert_close(old(pack_geometry(g)), new(pack_inputs(g, x)), atol=0, rtol=0)
        new.image_encoder[-1].weight.normal_(0, .1)
        assert not torch.equal(new(pack_inputs(g, x)), new(pack_inputs(g, np.zeros_like(x))))
    with pytest.raises(ValueError): new(dict(pack_inputs(g, x), future_endpoint=torch.zeros(4, 2)))


def test_normalizer_uses_only_source_fitting_past_rows_and_source_balances():
    x = np.arange(24*46, dtype=np.float32).reshape(24, 46)
    fold = dict(train_ids=np.arange(12), held_ids=np.arange(12, 24),
        groups=[np.arange(4), np.arange(4, 12)])
    a = normalizer(x, fold); x[12:] = 1e9; b = normalizer(x, fold)
    for k in a: np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_allclose(a['mean'], .5*x[:4].mean(0)+.5*x[4:12].mean(0))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.randn(12, 2)*.01)
    def forward(self, inputs): return inputs['baseline']+self.w


def test_training_engine_raw_control_and_actual_resume_match(tmp_path):
    g = geometry(12); y = np.ones((12, 12, 2), np.float32)
    valid = np.ones((12, 12), bool); valid[2] = False; y[2] = np.nan
    d = dict(geometry=g, target=y, valid=valid, sites=np.repeat(['a', 'b', 'c'], 4),
        motion=np.ones((12, 46), np.float32), quality=np.zeros((12, 46), np.float32))
    fold = dict(train_ids=np.arange(8), held_ids=np.arange(8, 12),
        groups=[np.arange(4), np.arange(4, 8)], factors=np.r_[np.ones(8), np.zeros(4)])
    settings = dict(steps=4, batch_size=6, learning_rate=.001, weight_decay=.0001,
        minimum_lr_ratio=.01, checkpoint_every=2, heartbeat_every=1, gradient_clip=5.)
    def run(name, legacy=False, resume=False, stop_at=None):
        torch.manual_seed(17); model = Tiny(); home = tmp_path/name
        kw = dict(seed=17, settings=settings, identity={'test': 'fixed'}, directory=home,
            resume=resume, stop_at=stop_at, heartbeat=lambda **_: None)
        if legacy: fit_trial(model, d, fold, **kw)
        else: fit(model, d, fold, arm='raw_control', **kw)
        return torch.load(home/'checkpoint.pt', weights_only=False)
    old = run('old', legacy=True); new = run('new')
    from scripts.run_m3w_moving_source_forecast import exact
    for k in old:
        if k != 'seconds': exact(old[k], new[k])
    run('resumed', stop_at=2); resumed = run('resumed', resume=True)
    for k in new:
        if k != 'seconds': exact(new[k], resumed[k])
    assert not resumed['draws'][8:].any()


def test_packing_rejects_nonfinite_context_without_future_or_label_parameters():
    import inspect
    assert list(inspect.signature(pack_inputs).parameters) == ['geometry', 'context']
    with pytest.raises(ValueError): pack_inputs(geometry(), np.full((4, 46), np.nan))
