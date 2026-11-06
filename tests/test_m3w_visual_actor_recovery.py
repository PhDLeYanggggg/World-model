import numpy as np
import pytest
import torch
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_visual_forecast import VisualSourceForecaster, pack_inputs
from tests.test_m3w_visual_forecast import ARCH, geometry


def test_actual_raw_core_training_with_frozen_image_branch_matches_old_updates():
    torch.manual_seed(17); old = SourceForecaster(ARCH, 1)
    torch.manual_seed(17); new = VisualSourceForecaster(ARCH, 1, np.zeros(46), np.ones(46), 'raw_control')
    a, b = (torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for m in (old, new))
    g = geometry(4); old.train(); new.train()
    for _ in range(3):
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        x, y = old(pack_geometry(g)), new(pack_inputs(g, np.zeros((4, 46), np.float32)))
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        x.square().mean().backward(); y.square().mean().backward()
        ga = torch.nn.utils.clip_grad_norm_(old.parameters(), 5.)
        gb = torch.nn.utils.clip_grad_norm_(new.parameters(), 5.)
        assert torch.equal(ga, gb)
        a.step(); b.step()
        for k, v in old.state_dict().items(): assert torch.equal(v, new.state_dict()[k])
    from scripts.run_m3w_moving_source_forecast import exact
    exact(a.state_dict()['state'], b.state_dict()['state'])


def test_cache_recovery_does_not_overwrite_complete_arrays(tmp_path):
    from scripts.run_m3w_moving_visual_actor import atomic_array
    path = tmp_path/'x.npy'; value = np.arange(12).reshape(3, 4)
    atomic_array(path, value); stamp = path.stat().st_mtime_ns
    atomic_array(path, value.copy()); assert path.stat().st_mtime_ns == stamp
    with pytest.raises(AssertionError): atomic_array(path, value+1)
    np.testing.assert_array_equal(np.load(path), value)


def test_completed_model_receipt_reused_without_retraining(tmp_path, monkeypatch):
    from scripts import run_m3w_moving_visual_actor as run
    path = tmp_path/'done.json'; checkpoint = tmp_path/'checkpoint.pt'
    identity = dict(example='fixed', arm='motion_geometry')
    torch.save(dict(identity=identity, step=4000, image_arm='motion_geometry'), checkpoint)
    receipt = dict(identity=identity, checkpoint={'path': str(checkpoint)}, result=dict(complete=True, step=4000))
    path.write_text(run.source.json.dumps(receipt))
    monkeypatch.setattr(run, 'checked', lambda ref: checkpoint)
    assert run.completed_fit(path, identity) == receipt
    with pytest.raises(ValueError): run.completed_fit(path, dict(example='changed'))
