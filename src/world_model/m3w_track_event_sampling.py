"""Train-only sampling contrasts over the immutable offline forecasting inputs."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from src.world_model.m3w_objective_alignment import geometry_prediction, objective_loss

MODES = ('row_uniform', 'scene_uniform', 'scene_track', 'scene_event_track')
EVENT_NAMES = ('static_stays', 'static_moves', 'moving_stops', 'moving_turns', 'other_motion')


def training_event_labels(geometry, targets):
    """Supervised sampling categories, never an inference feature or evaluation endpoint."""
    geometry, targets = np.asarray(geometry), np.asarray(targets)
    if (geometry.ndim != 2 or geometry.shape[1] < 16 or targets.shape != (len(geometry), 12, 2)
            or not np.isfinite(geometry).all() or not np.isfinite(targets).all()):
        raise ValueError('Finite registered past features and complete training labels required')
    past = geometry[:, :16].reshape(-1, 8, 2)
    static = np.all(past == 0, axis=(1, 2))
    future_static = np.all(targets == 0, axis=(1, 2))
    late_static = np.all(targets[:, -4:] == targets[:, -1:, :], axis=(1, 2))
    a, b = past[:, -1]-past[:, -4], targets[:, -1]-targets[:, -4]
    denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    cosine = np.divide((a*b).sum(1), denominator, out=np.ones(len(a)), where=denominator > 0)
    turn = (denominator > 0) & (cosine <= np.cos(np.pi/4))
    result = np.full(len(past), 4, np.int64)
    result[~static & turn] = 3
    result[~static & late_static] = 2
    result[static & ~future_static] = 1
    result[static & future_static] = 0
    return result


def sampling_probabilities(scenes, tracks, events, mode):
    scenes, tracks, events = map(np.asarray, (scenes, tracks, events))
    n = len(scenes)
    if (not n or scenes.shape != (n,) or tracks.shape != (n,) or events.shape != (n,)
            or mode not in MODES or not np.isin(events, np.arange(len(EVENT_NAMES))).all()):
        raise ValueError('Nonempty aligned training groups and a registered sampler required')
    for key in np.unique(tracks):
        if len(np.unique(scenes[tracks == key])) != 1:
            raise ValueError('Recording-local track must belong to one physical scene')
    if mode == 'row_uniform':
        return np.ones(n, np.float64)/n
    p = np.zeros(n, np.float64)
    unique_scenes = np.unique(scenes)
    for scene in unique_scenes:
        members = np.flatnonzero(scenes == scene)
        scene_mass = 1/len(unique_scenes)
        if mode == 'scene_uniform':
            p[members] = scene_mass/len(members)
            continue
        categories = np.unique(events[members]) if mode == 'scene_event_track' else [None]
        for event in categories:
            group = members if event is None else members[events[members] == event]
            identities = np.unique(tracks[group])
            for key in identities:
                rows = group[tracks[group] == key]
                p[rows] = scene_mass/len(categories)/len(identities)/len(rows)
    return p/p.sum()


def train_distribution(indices, folds, tracks, geometry, targets, mode):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(folds)):
        raise ValueError('Unique valid training indices required')
    # Slice before labeling: held labels cannot even affect category construction.
    labels = training_event_labels(geometry[indices], targets[indices])
    probabilities = sampling_probabilities(np.asarray(folds)[indices], np.asarray(tracks)[indices], labels, mode)
    return probabilities, labels


def distribution_summary(probabilities, scenes, tracks, events, draw_counts=None):
    p = np.asarray(probabilities)
    result = dict(rows=len(p), distinct_recording_local_tracks=len(np.unique(tracks)),
                  inverse_squared_mass_diagnostic=float(1/np.sum(p*p)),
                  inverse_squared_mass_is_not_independent_sample_size=True)
    for name, group in (('scene', scenes), ('event', events)):
        result[name+'_mass'] = {str(v):float(p[np.asarray(group) == v].sum()) for v in np.unique(group)}
        result[name+'_row_count'] = {str(v):int(np.sum(np.asarray(group) == v)) for v in np.unique(group)}
        if draw_counts is not None:
            result[name+'_draw_count'] = {str(v):int(np.asarray(draw_counts)[np.asarray(group) == v].sum())
                                         for v in np.unique(group)}
    return result


def fit_sampled(model, batch, train_ids, probabilities, *, mode, config, seed,
                identity, checkpoint, heartbeat, stop_at=None):
    """Versioned trainer: same log-ADE loss and optimizer, explicit sampling mass."""
    weights = torch.as_tensor(probabilities, dtype=torch.float64)
    if (mode not in MODES or weights.shape != train_ids.shape or not torch.isfinite(weights).all()
            or not (weights > 0).all() or abs(float(weights.sum())-1) > 1e-10):
        raise ValueError('Positive normalized train-only sampling probabilities required')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed+7919)
    step, losses, elapsed = 0, [], 0.
    counts = torch.zeros(len(train_ids), dtype=torch.int64)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['config'] != config or state['mode'] != mode
                or not torch.equal(state['probabilities'], weights) or not torch.equal(state['train_ids'], train_ids)):
            raise ValueError('Changed resume identity, data membership or sampling distribution')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        step, losses, elapsed, counts = state['step'], state['losses'], state['fit_seconds'], state['draw_counts']
    initial = step
    limit = min(config['updates'], stop_at) if stop_at is not None else config['updates']
    started = time.monotonic()
    model.train()
    while step < limit:
        selected = torch.multinomial(weights, config['batch_size'], replacement=True, generator=generator)
        counts += torch.bincount(selected, minlength=len(train_ids))
        x, observed, baseline, target = batch(train_ids[selected])
        optimizer.zero_grad(set_to_none=True)
        loss, detail = objective_loss(geometry_prediction(model, x, observed, baseline), target, baseline, 'log')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step == 1 or step % 100 == 0:
            losses.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), **detail))
        if step % config['checkpoint_every'] == 0 or step == limit:
            seconds = elapsed+time.monotonic()-started
            state = dict(identity=identity, config=config, mode=mode, model=model.state_dict(),
                optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                step=step, losses=losses, fit_seconds=seconds, probabilities=weights, train_ids=train_ids,
                draw_counts=counts)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(state, temporary)
            os.replace(temporary, checkpoint)
            heartbeat(dict(state='fit_complete' if step == config['updates'] else 'training',
                           step=step, fit_seconds=seconds, loss=float(loss.detach())))
    model.eval()
    return dict(step=step, new_updates_this_invocation=step-initial, losses=losses,
        fit_seconds=elapsed+time.monotonic()-started, complete=step == config['updates'],
        draw_counts=counts.numpy())
