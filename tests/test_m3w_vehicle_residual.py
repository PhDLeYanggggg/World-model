import numpy as np
import torch

from src.world_model import m3w_vehicle_residual as model


def test_completion_is_observed_loss_plus_exact_missing_step_debit():
    cv = torch.zeros((1, 12, 2))
    p = torch.zeros_like(cv)
    p[0, :, 0] = 2.
    target = torch.zeros_like(cv)
    mask = torch.zeros((1, 12), dtype=torch.bool)
    mask[0, :8] = True
    target[0, 8:] = float('nan')
    torch.testing.assert_close(model.loss(p, cv, target, mask, completion=False), torch.tensor(16/12))
    torch.testing.assert_close(model.loss(p, cv, target, mask, completion=True), torch.tensor(2.))
    mask[:] = True
    target = torch.zeros_like(cv)
    torch.testing.assert_close(model.loss(p, cv, target, mask, completion=False),
                               model.loss(p, cv, target, mask, completion=True))


def test_prediction_respects_peak_bound_and_gradients_are_finite():
    h = np.random.default_rng(37).normal(size=(10, 8, 2)).cumsum(axis=1)
    x, cv, _, _ = model.encode(h)
    net = model.motion.network(17)
    with torch.no_grad():
        net[-1].bias.fill_(5.)
    y = model.predict(net, torch.from_numpy(x), torch.from_numpy(cv))
    peak = (y-torch.from_numpy(cv)).norm(dim=-1).amax()
    assert peak < model.PEAK_BOUND + 1e-5
    y.square().mean().backward()
    assert all(torch.isfinite(x.grad).all() for x in net.parameters())


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    from scripts import run_m3w_opendd_vehicle_residual as run

    rng = np.random.default_rng(11)
    h = rng.normal(size=(32, 8, 2)).cumsum(axis=1)
    x, cv, _, _ = model.encode(h)
    y = (cv + rng.normal(size=cv.shape)*.1).astype(np.float32)
    tensors = tuple(map(torch.from_numpy, (x, cv, y, np.ones(cv.shape[:2], dtype=bool))))
    data = {s: dict(tensors=tensors) for s in run.SITES}
    monkeypatch.setattr(run, 'beat', lambda *a, **k: None)
    monkeypatch.setattr(run, 'BATCH', 8)
    monkeypatch.setattr(run, 'SAVE', 2)
    a, b = tmp_path/'resume', tmp_path/'full'
    a.mkdir()
    b.mkdir()
    monkeypatch.setattr(run, 'LOCAL', a)
    monkeypatch.setattr(run, 'STEPS', 2)
    run.fit('fixture', data, 'rdb1', 17, 'completion')
    monkeypatch.setattr(run, 'STEPS', 4)
    run.fit('fixture', data, 'rdb1', 17, 'completion')
    monkeypatch.setattr(run, 'LOCAL', b)
    run.fit('fixture', data, 'rdb1', 17, 'completion')
    name = 'rdb1_17_completion.pt'
    state = [torch.load(p/name, weights_only=True) for p in (a, b)]
    for k in state[0]['weights']:
        torch.testing.assert_close(state[0]['weights'][k], state[1]['weights'][k], atol=0., rtol=0.)


def test_publisher_freight_classes_are_used_but_separate_trailers_are_not():
    import sqlite3

    from scripts import run_m3w_opendd_vehicle_residual as run

    conn = sqlite3.connect(':memory:')
    try:
        conn.execute('CREATE TABLE rdb1_1 (OBJID INT, TIMESTAMP REAL, UTM_X REAL, UTM_Y REAL, CLASS TEXT)')
        classes = ('Car', 'Medium Vehicle', 'Heavy Vehicle', 'Motorcycle', 'Bus', 'Trailer', 'Pedestrian')
        values = [(ident, i*12*1001./30000., 400000.+i, 5500000.+ident, c)
                  for ident, c in enumerate(classes) for i in range(20)]
        conn.executemany('INSERT INTO rdb1_1 VALUES (?, ?, ?, ?, ?)', values)
        d, stats = run.extract_flight(conn, 'rdb1_1')
        np.testing.assert_array_equal(d['agent'], np.arange(5))
        assert stats['Trailer'] == 20 and d['valid'].all()
    finally:
        conn.close()
