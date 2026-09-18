"""Matched past-RGB / coverage-only classifiers, not trajectory policies."""
import os
import time

import numpy as np
import torch
from torch import nn

ARMS = ('mask_only', 'past_rgb')


class VisualStart(nn.Module):
    def __init__(self, width=476):
        super().__init__()
        self.width = width
        self.image = nn.Sequential(
            nn.Conv2d(4, 8, 3, 2, 1), nn.SiLU(), nn.Conv2d(8, 16, 3, 2, 1),
            nn.SiLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64, 16), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(width+8+128, 64), nn.SiLU(),
                                  nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, geometry, rgb, coverage, arm):
        n = len(geometry)
        if (arm not in ARMS or geometry.shape != (n, self.width)
                or rgb.shape != (n, 8, 3, 32, 32) or coverage.shape != (n, 8, 1, 32, 32)):
            raise ValueError('Fixed past geometry and eight-frame image schema required')
        rgb = torch.where(coverage > 0, rgb, torch.zeros_like(rgb))
        if arm == 'mask_only':
            rgb = torch.zeros_like(rgb)
        image = torch.cat((rgb, coverage), 2).flatten(0, 1)
        visual = self.image(image).reshape(n, 8, 16)
        visual = visual * (coverage.sum((2, 3, 4)) > 0)[..., None]
        features = torch.cat((geometry, coverage.mean((2, 3, 4)), visual.flatten(1)), 1)
        return self.head(features).squeeze(-1)


def fit_visual(model, inputs, labels, train_ids, weights, *, arm, seed, config,
               identity, checkpoint, heartbeat, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Fixed image arm required')
    ids = np.asarray(train_ids, dtype=int)
    w = torch.as_tensor(weights, dtype=torch.float64)
    if (not len(ids) or len(np.unique(ids)) != len(ids) or w.shape != (len(ids),)
            or torch.any(w <= 0) or not torch.isclose(w.sum(), torch.tensor(1., dtype=w.dtype))):
        raise ValueError('Unique training rows and normalized positive weights required')
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed+7919)
    step, seconds, losses = 0, 0., []
    draws = np.zeros(len(ids), np.int64)
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if cp['identity'] != identity or cp['config'] != config or cp['arm'] != arm:
            raise ValueError('Checkpoint identity/config/arm changed')
        np.testing.assert_array_equal(cp['train_ids'], ids)
        model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer'])
        generator.set_state(cp['sampler_rng']); torch.set_rng_state(cp['torch_rng'])
        step, seconds, losses, draws = cp['step'], cp['fit_seconds'], cp['losses'], cp['draw_counts']
    first = step; started = time.monotonic()
    limit = config['updates'] if stop_at is None else min(stop_at, config['updates'])
    if min(limit, config['batch_size'], config['checkpoint_every']) <= 0:
        raise ValueError('Positive budgets required')
    model.train()
    try:
        while step < limit:
            local = torch.multinomial(w, config['batch_size'], replacement=True, generator=generator).numpy()
            batch_ids = ids[local]
            features = inputs(batch_ids)
            y = labels(batch_ids)
            opt.zero_grad(set_to_none=True)
            logit = model(*features, arm)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite visual classifier loss')
            loss.backward(); grad = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step(); step += 1
            np.add.at(draws, local, 1)
            if step == 1 or step % 100 == 0:
                losses.append(dict(step=step, bce=float(loss.detach()), gradient_norm=float(grad)))
            if step == limit or step % config['checkpoint_every'] == 0:
                value = dict(identity=identity, config=config, arm=arm, model=model.state_dict(),
                    optimizer=opt.state_dict(), sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                    step=step, fit_seconds=seconds+time.monotonic()-started, losses=losses,
                    train_ids=ids, draw_counts=draws)
                checkpoint.parent.mkdir(parents=True, exist_ok=True); tmp = checkpoint.with_suffix('.tmp')
                torch.save(value, tmp); os.replace(tmp, checkpoint)
                heartbeat(state='training', step=step, bce=float(loss.detach()), fit_seconds=value['fit_seconds'])
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_completed_step=step)
        raise
    model.eval()
    return dict(step=step, complete=step == config['updates'], new_updates=step-first,
        fit_seconds=seconds+time.monotonic()-started, losses=losses,
        distinct_sampled_rows=int((draws > 0).sum()), total_draws=int(draws.sum()))


def probabilities(model, inputs, ids, arm):
    model.eval(); result = []
    with torch.no_grad():
        for offset in range(0, len(ids), 128):
            result.append(model(*inputs(np.asarray(ids[offset:offset+128])), arm).sigmoid().numpy())
    return np.concatenate(result)
