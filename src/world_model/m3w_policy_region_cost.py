"""Fitting-only policy-region moment loss; unchanged causal inference model."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from src.world_model import m3w_multiaction_cost as base
from src.world_model.m3w_scene_context_cost import SceneCostHead
from src.world_model.m3w_native_forecast import draw_batch

MOMENTS = (1, 2, 3, 4)


def prepare_regions(labels, scales, sites, recordings, choices, *, arm):
    y, scales, sites, records, choices = map(np.asarray, (labels, scales, sites, recordings, choices))
    n = len(choices)
    if (y.shape != (n, 7, 5) or scales.shape != (n,) or sites.shape != (n,)
            or records.shape != (n,) or (scales <= 0).any() or not np.isfinite(scales).all()):
        raise ValueError('Aligned original fitting labels, scales and teacher choices required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all():
        raise ValueError('Unknown fitting outcomes remain unknown')
    actions = list(base.ARMS[arm]); active = known & (choices >= 0)
    if not np.isin(choices[choices >= 0], actions).all():
        raise ValueError('Teacher choice outside the legal bank')
    pool = np.full(n, -1, np.int64); local = np.full(n, -1, np.int64)
    keys, targets = [], []
    for site, recording in sorted(set(zip(sites[active].tolist(), records[active].astype(str).tolist()))):
        ix = np.flatnonzero(active & (sites == site) & (records.astype(str) == recording))
        target = y[ix, choices[ix]]/scales[ix, None]
        pool[ix] = len(keys); local[ix] = [actions.index(int(k)) for k in choices[ix]]
        keys.append((str(site), str(recording))); targets.append(target.mean(0))
    return dict(pool=torch.from_numpy(pool), local_action=torch.from_numpy(local),
        target=torch.tensor(np.asarray(targets, np.float32).reshape(-1, 5)), keys=keys,
        selected_known_rows=int(active.sum()), unknown_selected_rows=int((~known & (choices >= 0)).sum()),
        record_ids_are_training_loss_metadata_not_model_inputs=True)


def region_loss(prediction, row_ids, regions, rms):
    """Stochastic conditional-moment matching, not pointwise deviance or calibration proof."""
    pool = regions['pool'][row_ids]; local = regions['local_action'][row_ids]
    at = torch.nonzero(pool >= 0, as_tuple=False).flatten()
    if not len(at):
        return prediction.sum()*0
    unique, inverse = torch.unique(pool[at], sorted=True, return_inverse=True)
    selected = prediction[at, local[at]][:, MOMENTS]
    sums = torch.zeros((len(unique), len(MOMENTS)), dtype=prediction.dtype, device=prediction.device)
    sums = sums.index_add(0, inverse, selected)
    count = torch.bincount(inverse, minlength=len(unique)).to(prediction.dtype)[:, None]
    predicted = (sums/count)/rms[list(MOMENTS)]
    truth = regions['target'][unique][:, MOMENTS].detach()/rms[list(MOMENTS)]
    safe = predicted+1e-12
    divergence = safe-truth+torch.xlogy(truth, truth)-truth*torch.log(safe)
    if not torch.isfinite(divergence).all():
        raise FloatingPointError('Nonfinite region-moment objective')
    return divergence.mean()


def fit(data, *, arm, settings, seed, identity, path, regions, region_weight,
        heartbeat=lambda **_:None, guard=lambda:None, resume=False, stop_at=None):
    if arm not in base.ARMS or region_weight < 0:
        raise ValueError('Fixed bank and nonnegative registered region weight required')
    pr = data['preprocess']; actions = torch.tensor(base.ARMS[arm])
    model = SceneCostHead(data['z'].shape[-1], settings['width'], seed, pr['bias'])
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    binding = dict(arm=arm, settings=settings, seed=seed, identity=identity, preprocess=pr,
        input_hashes=data['input_hashes'], initial_model=initial, dimension=data['z'].shape[-1],
        parameters=sum(p.numel() for p in model.parameters()), region_weight=region_weight,
        region_keys=regions['keys'], selected_known_region_rows=regions['selected_known_rows'],
        region_arrays={key:hashlib.sha256(regions[key].numpy().tobytes()).hexdigest()
                       for key in ('pool', 'local_action', 'target')})
    step, elapsed, trace, draw_hash, row_draws = 0, 0., [], '0'*64, 0
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError('Explicit same-checkpoint resume required')
        state = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(state[key], value)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, elapsed, trace, draw_hash, row_draws = (state[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit == 0:
        raise ValueError('Fixed nondecreasing optimizer budget required')
    started = time.monotonic(); rms = torch.from_numpy(pr['rms'])
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, model=model.state_dict(),
            optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            draw_hash=draw_hash, row_draws=row_draws, unknown_rows_sampled=0,
            inner_held_used_for_training=False))
    if step == 0:
        save()
    while step < limit:
        ix = draw_batch(data['groups'], settings['batch_size'], rng)
        prediction = model(data['z'][ix][:, actions], data['envelope'][ix][:, actions], actions)
        terms = base.continuous_loss(prediction, data['truth'][ix][:, actions], data['allowed'][ix][:, actions], rms,
            underharm_weight=settings['underharm_weight'], ranking_weight=settings['ranking_weight'])
        extra = region_loss(prediction, ix, regions, rms) if region_weight else None
        total = terms['total'] if not region_weight else terms['total']+region_weight*extra
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite fixed region-cost objective')
        optimizer.zero_grad(set_to_none=True); total.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; row_draws += len(ix)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(total.detach()), regression=float(terms['regression'].detach()),
                ranking=float(terms['ranking'].detach()), gradient_norm=float(norm))
            if region_weight:
                row['region_moment_loss'] = float(extra.detach())
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return base.read_checkpoint(path)


def assert_neutral(reference, state):
    for key in reference:
        if key not in ('identity', 'seconds'):
            base.exact(reference[key], state[key])
    if state['region_weight'] != 0:
        raise ValueError('Disabled loss must have exactly zero weight')
