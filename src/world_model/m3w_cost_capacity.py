"""Explicit continuation contracts and immutable prefix checkpoints."""
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from src.world_model.m3w_tempered_cost_head import fit
from src.evaluation.m3w_experiment_contract import file_digest


def fork_continuation(source, destination, *, identity, settings, seed, prefix_steps):
    source, destination = Path(source), Path(destination)
    if destination.exists(): raise ValueError('Continuation destination already exists')
    before = file_digest(source)
    cp = torch.load(source, map_location='cpu', weights_only=False)
    if (cp['step'] != prefix_steps or cp['seed'] != seed or cp['loss_exponent'] != 1
            or cp['forward_arm'] != 'bounded_native' or settings['steps'] <= prefix_steps
            or {k:v for k,v in cp['settings'].items() if k != 'steps'} !=
               {k:v for k,v in settings.items() if k != 'steps'}):
        raise ValueError('Exact prefix optimizer, architecture, sampling and loss must be preserved')
    # Only the new run contract and newly spent time change, never parent weights.
    cp = dict(cp, identity=identity, settings=settings, seconds=0.)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp'); torch.save(cp, temporary); os.replace(temporary, destination)
    assert file_digest(source) == before
    return before


def save_prefix(checkpoint, prefix, *, steps, identity):
    checkpoint, prefix = Path(checkpoint), Path(prefix)
    if prefix.exists():
        cp = torch.load(prefix, map_location='cpu', weights_only=False)
        if cp['step'] != steps or cp['identity'] != identity: raise ValueError('Changed prefix contract')
        return
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if cp['step'] != steps or cp['identity'] != identity: raise ValueError('Cannot reconstruct a missing earlier prefix')
    temporary = prefix.with_suffix('.tmp'); shutil.copyfile(checkpoint, temporary); os.replace(temporary, prefix)


def fit_path(x, y, distance, sites, pr, *, width, source, settings, identity, seed,
             directory, heartbeat, resume=False, stop_at=None, prefix_steps=3000):
    directory = Path(directory); checkpoint = directory/'checkpoint.pt'; prefix = directory/'prefix.pt'
    if checkpoint.exists() and not resume: raise ValueError('Existing path requires explicit resume')
    inherited = prefix_steps if width == 'narrow' else 0
    if stop_at is not None and not inherited < stop_at <= settings['steps']:
        raise ValueError('Pilot must advance the inherited prefix within the fixed budget')
    if width not in ('narrow', 'wide'): raise ValueError('Registered width family required')
    if width == 'narrow' and not checkpoint.exists():
        fork_continuation(source, checkpoint, identity=identity, settings=settings, seed=seed, prefix_steps=prefix_steps)
    current = 0 if not checkpoint.exists() else torch.load(checkpoint, map_location='cpu', weights_only=False)['step']
    limit = settings['steps'] if stop_at is None else stop_at
    if limit < current: raise ValueError('Cannot run backwards')
    common = dict(seed=seed, settings=settings, identity=identity, directory=directory, heartbeat=heartbeat)
    if width == 'wide' and current < prefix_steps:
        _, result = fit(x, y, distance, sites, pr, resume=checkpoint.exists(), stop_at=min(limit, prefix_steps), **common)
        current = result['step']
        if current < prefix_steps: return result
    if width == 'wide': save_prefix(checkpoint, prefix, steps=prefix_steps, identity=identity)
    _, result = fit(x, y, distance, sites, pr, resume=True, stop_at=limit, **common)
    result['inherited_updates'] = inherited
    result['newly_trained_updates'] = result['step']-inherited
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    reference = torch.load(source, map_location='cpu', weights_only=False)
    prefix_cp = reference if width == 'narrow' else torch.load(prefix, map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(prefix_cp['draws'], reference['draws'])
    assert cp['draws'].sum() == result['step']*settings['batch_size']
    assert cp['draws'][~pr['known']].sum() == 0
    return result
