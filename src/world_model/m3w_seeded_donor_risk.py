"""Seed-explicit replication of the frozen donor risk training algorithm."""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_donor_augmentation as original

SEEDS = (17, 29, 43)


def fit(data, source, identity, path, *, seed, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only the three predeclared risk-head seeds')
    parent = original.parent
    p = parent.prepare(data, source)
    cfg = dict(original.SETTINGS, seed=seed)
    torch.manual_seed(seed)
    model = parent.Head(cfg)
    nn.init.zeros_(model.net[-1].weight)
    with torch.no_grad():
        model.net[-1].bias.copy_(torch.from_numpy(p['prior']))
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = torch.Generator().manual_seed(seed)
    prob = p['w'].double()/p['w'].double().sum()
    n = len(prob)
    binding = dict(identity=identity, settings=cfg, preprocess=p['binding'], initial_model=initial,
                   parameters=sum(v.numel() for v in model.parameters()), fit_rows=n)
    step, seconds, trace = 0, 0., []
    visits = np.zeros(n, np.int32)
    path = Path(path)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        saved = parent.storage.base.read_checkpoint(path)
        for k, v in binding.items():
            parent.storage.base.exact(saved[k], v)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, visits = (saved[k] for k in ('step', 'seconds', 'trace', 'visits'))
    limit = cfg['steps'] if stop_at is None else stop_at
    if not step <= limit <= cfg['steps']:
        raise ValueError('Fixed nondecreasing budget required')
    start = time.monotonic()

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        parent.storage.base.save_checkpoint(path, dict(**binding, model=model.state_dict(),
            optimizer=opt.state_dict(), generator=rng.get_state(), torch_rng=torch.get_rng_state(),
            step=step, seconds=seconds+time.monotonic()-start, trace=trace,
            visits=visits, complete=step == cfg['steps']))

    try:
        while step < limit:
            guard()
            ix = torch.multinomial(prob, cfg['batch_size'], replacement=True, generator=rng)
            opt.zero_grad(set_to_none=True)
            loss = ((model(p['x'][ix])-p['y'][ix])**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite risk loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            step += 1
            np.add.at(visits, ix.numpy(), 1)
            if step == 1 or step % cfg['log_every'] == 0:
                row = dict(step=step, batch_loss=float(loss.detach()), gradient_norm=float(norm),
                           distinct_fitting_rows_drawn=int((visits > 0).sum()))
                trace.append(row)
                heartbeat(state='seeded_donor_training', **row)
            if step % cfg['checkpoint_steps'] == 0 or step == limit:
                save()
    except BaseException:
        save()
        raise
    return parent.storage.base.read_checkpoint(path)


def bootstrap_costs(costs, reference, *, resamples=3000, seed=62027):
    """Paired source-cluster sampling, shared across risk seeds and both arms."""
    if (resamples != 3000 or len(costs) != 3 or len(reference) != 3
            or any(np.shape(a) != np.shape(b) or np.shape(a) not in ((3, 2), (3, 3))
                   for a, b in zip(costs, reference))
            or any(not np.isfinite(a).all() or np.any(a < 0) for a in (*costs, *reference))):
        raise ValueError('Three risk seeds, three source-grouped contexts and3000replicates')
    generator = np.random.default_rng(seed)
    effect = []
    for _ in range(resamples):
        aa, bb = [], []
        for a, b in zip(costs, reference):
            pick = generator.integers(0, a.shape[1], a.shape[1])
            aa.append(a[:, pick].mean())
            bb.append(b[:, pick].mean())
        denominator = float(np.mean(bb))
        effect.append(100*(1-float(np.mean(aa))/denominator) if denominator > 0 else np.nan)
    values = np.asarray(effect)
    finite = np.isfinite(values)
    return dict(resamples=resamples, seed=seed, valid=int(finite.sum()),
        interval95=np.quantile(values[finite], [.025, .975]).tolist() if finite.any() else None,
        unit='conservative_source_component_not_window', risk_seeds_resampled=False,
        seed_averaged_before_source_cluster_effect=True, independent_confirmation=False)
