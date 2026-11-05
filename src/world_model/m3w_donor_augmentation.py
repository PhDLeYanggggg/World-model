"""Fixed-budget, source-disjoint donor augmentation for the original risk task."""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_neural_risk_transport as parent
from src.world_model import m3w_rollout_history_risk as relation
from src.world_model import m3w_candidate_neighbor_basis as neighbor
from src.world_model import m3w_shared_motion_cost as causal_api

ARMS = ('own_only', 'A_only', 'augmented')
SETTINGS = dict(parent.SETTINGS, steps=4096, log_every=256)


def causal_features(geometry, shared_context, physical, proposal, recordings, frames, ids, sites, floor):
    """No outcomes or label-validity masks can enter this feature interface."""
    paths = np.concatenate((physical, proposal[:, None]), 1)
    c = causal_api.features(geometry=geometry, shared_context=shared_context,
        rollouts=paths, floor_index=floor, recordings=recordings, frames=frames, ids=ids, sites=sites)
    x = np.concatenate((relation.parent.kin.motion_basis(c['x']), relation.basis(c['x']),
                        neighbor.basis(c['x'])), -1)[:, 6]
    if x.shape != (len(ids), 120) or not np.isfinite(x).all():
        raise ValueError('Finite original120 only')
    return x, c['scales'], c['allowed'][:, 6], paths


def supervision(paths, target, valid, scale, floor, easy_cut):
    """Match the old float32 signed risk targets; unknown labels stay unknown."""
    m, known = causal_api.base.base.labels(paths, target, valid, floor, easy_cut)
    norm = (m / scale[:, None, None]).astype(np.float32)
    # The reference implementation applies the risk subtraction in Torch float32.
    t = torch.from_numpy(norm[:, 6])
    y = (t[:, [1, 4]] - .02*t[:, [2, 3]]).numpy()
    return y, known


def combined(own, added, source, arm):
    if arm not in ARMS or set(own['sites']) & set(added['sites']):
        raise ValueError('Disjoint parent-A and own-B groups required')
    fit = own['sites'] != source
    if not fit.any() or fit.all() or source in set(added['sites']):
        raise ValueError('Query source must be entirely excluded from donors')
    ow = np.zeros(len(fit), float)
    if arm != 'A_only':
        ow[fit] = own['weights'][fit]/own['weights'][fit].sum()
    aw = added['weights']/added['weights'].sum() if arm != 'own_only' else np.zeros(len(added['sites']))
    if arm == 'augmented':
        ow *= .5
        aw *= .5
    # Zero-weight rows are removed, not sampled and not used in normalization.
    keep_o = (ow > 0) | ~fit
    keep_a = aw > 0
    result = {}
    for k in ('x', 'target', 'sites'):
        result[k] = np.concatenate((own[k][keep_o], added[k][keep_a]))
    result['ids'] = np.concatenate((own['ids'][keep_o]*7, added['ids'][keep_a]))
    result['weights'] = np.concatenate((ow[keep_o], aw[keep_a]))
    if len(set(result['ids'])) != len(result['ids']):
        raise ValueError('Unique original-row/action donor namespace required')
    return result


def fit(data, source, identity, path, *, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    p = parent.prepare(data, source)
    cfg = SETTINGS
    torch.manual_seed(cfg['seed'])
    model = parent.Head(cfg)
    nn.init.zeros_(model.net[-1].weight)
    with torch.no_grad():
        model.net[-1].bias.copy_(torch.from_numpy(p['prior']))
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = torch.Generator().manual_seed(cfg['seed'])
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
                heartbeat(state='risk_augmentation_training', **row)
            if step % cfg['checkpoint_steps'] == 0 or step == limit:
                save()
    except BaseException:
        save()
        raise
    return parent.storage.base.read_checkpoint(path)


def exact(a, b):
    for k in ('identity', 'settings', 'preprocess', 'initial_model', 'model', 'optimizer',
              'generator', 'torch_rng', 'step', 'trace', 'visits', 'complete', 'fit_rows'):
        parent.storage.base.exact(a[k], b[k])
