"""Refresh fit-only selection emphasis without changing data, draws or inference."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_conditional_cost_head import region_weights, loss_value
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized
from src.evaluation.m3w_conditional_cost_audit import strict_bits


def refreshed_region(model, x, distance, past, pr, multiplier):
    """No future targets or held-source rows enter the selection rule."""
    was_training = model.training
    score = predict(model, x, distance, pr, 'bounded_native')
    model.train(was_training)
    bits = strict_bits(score, past, distance)
    return bits, region_weights(bits, pr, multiplier)


def fit(x, y, distance, past, sites, pr, initial_weights, *, seed, settings,
        refresh_every, multiplier, identity, directory, heartbeat, resume=False, stop_at=None):
    initial_weights = np.asarray(initial_weights, np.float32)
    if (initial_weights.shape != (len(x),) or not np.isfinite(initial_weights).all()
            or (initial_weights[pr['known']] <= 0).any()
            or (initial_weights[~pr['known']] != 0).any()
            or not np.isclose(np.dot(pr['weights'], initial_weights), 1.)
            or past.shape != (len(x), 8, 2) or not np.isfinite(past).all()
            or not isinstance(refresh_every, int) or refresh_every <= 0
            or not np.isfinite(multiplier) or multiplier < 1):
        raise ValueError('Supported initial weights, causal history and fixed refresh schedule required')
    model = build(x.shape[1], settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    d = torch.from_numpy((distance/pr['cost_scale']).astype(np.float32))
    target = torch.from_numpy(np.where(pr['known'][:, None], y/pr['cost_scale'], 0).astype(np.float32))
    current_weights = initial_weights.copy()
    weights = torch.from_numpy(current_weights)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    path = Path(directory)/'checkpoint.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    refreshes, refresh_seconds = [], 0.
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if (cp['identity'] != identity or cp['settings'] != settings or cp['seed'] != seed
                or cp['loss_exponent'] != 1 or cp['forward_arm'] != 'bounded_native'
                or cp['refresh_every'] != refresh_every or cp['region_multiplier'] != multiplier):
            raise ValueError('Changed resume identity')
        np.testing.assert_array_equal(cp['initial_loss_weights'], initial_weights)
        for k in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(cp['preprocess'][k], pr[k])
        for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
            assert cp['preprocess'][k] == pr[k]
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng'])
        torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
        current_weights = cp['loss_weights'].copy()
        weights = torch.from_numpy(current_weights)
        refreshes, refresh_seconds = cp['refreshes'], cp['refresh_seconds']
        assert [r['step'] for r in refreshes] == list(range(refresh_every, step, refresh_every))
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit <= 0 or limit < step:
        raise ValueError('Nondecreasing fixed budget required')

    def save():
        cp = dict(identity=identity, settings=settings, seed=seed, loss_exponent=1,
            forward_arm='bounded_native', initial_loss_weights=initial_weights,
            loss_weights=current_weights, refresh_every=refresh_every, region_multiplier=multiplier,
            refreshes=refreshes, refresh_seconds=refresh_seconds, preprocess=pr,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), step=step, seconds=seconds+time.monotonic()-started,
            trace=trace, draws=draws)
        temp = path.with_suffix('.tmp')
        torch.save(cp, temp)
        os.replace(temp, path)

    try:
        model.train()
        while step < limit:
            # A checkpoint at the boundary refreshes once before its next draw.
            if step > 0 and step % refresh_every == 0:
                t = time.monotonic()
                bits, current_weights = refreshed_region(model, x, distance, past, pr, multiplier)
                weights = torch.from_numpy(current_weights)
                refresh_seconds += time.monotonic()-t
                refreshes.append(dict(step=step, selected=bits,
                    model={k: v.detach().clone() for k, v in model.state_dict().items()},
                    complete_selected=int((bits & pr['known']).sum()),
                    mean_weight=float(np.dot(pr['weights'], current_weights))))
                heartbeat(state='fit_region_refreshed', step=step, selected=int(bits.sum()),
                          complete_selected=refreshes[-1]['complete_selected'])
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            score = model(z[ids], d[ids], 'bounded_native')
            loss = loss_value(score, target[ids], d[ids], weights[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adaptive cost loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row)
                heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step)
        raise
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unknown_rows_sampled=int(draws[~pr['known']].sum()), trace=trace,
        refresh_seconds=refresh_seconds, refresh_count=len(refreshes),
        parameters=sum(p.numel() for p in model.parameters()))
