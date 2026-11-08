import numpy as np
import sqlite3
import torch

from scripts import run_m3w_opendd_velocity_candidate as runner
from src.world_model.m3w_opendd_velocity_candidate import decode, design, objective


def test_local_context_masks_missing_and_is_equivariant():
    h = np.stack([np.arange(8), np.zeros(8)],axis=1)[None].astype(float)
    peer = np.full((1,2,2,8,2),np.nan)
    peer[0,0,0] = h[0] + np.array([1.,2.])
    rotate = np.array([[0.,1.],[-1.,0.]])
    x,_,_ = design(h,peer)
    shifted,_,_ = design(h @ rotate + 100.,peer @ rotate + 100.)
    np.testing.assert_allclose(x,shifted,atol=1e-6)
    assert np.isfinite(x).all() and not x[:,-48:].any()


def test_velocity_knots_integrate_and_unknown_targets_do_not_change_loss():
    v = decode(torch.tensor([[1.,0.,1.,0.,1.,0.]],requires_grad=True))
    torch.testing.assert_close(v[0,:,0],torch.arange(1,13)*v[0,0,0])
    future = torch.zeros_like(v)
    valid = torch.ones((1,12),dtype=torch.bool)
    valid[0,-1] = False
    alt = future.clone()
    alt[0,-1] = float('nan')
    a = objective(v,future,valid,future)
    torch.testing.assert_close(a,objective(v,alt,valid,future))
    a.backward()


def test_vehicle_query_reads_only_requested_past_motor_positions(monkeypatch):
    monkeypatch.setattr(runner,'np',np,raising=False)
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE rdb1_1 (OBJID,TIMESTAMP,UTM_X,UTM_Y,CLASS)')
        conn.executemany('INSERT INTO rdb1_1 VALUES (?,?,?,?,?)',
            [(1,0.,2.,3.,'Car'),(1,.4004,4.,5.,'Car'),(2,0.,2.,3.,'Pedestrian')])
        got = runner.read_traffic(conn,'rdb1_1',[0])
    assert dict(got) == {0:{1:(2.,3.)}}
