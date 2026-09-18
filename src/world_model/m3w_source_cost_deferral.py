"""Training-only baseline-relative deferral with an exact-zero deterministic action."""
import copy
import time

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_source_cost_dynamics import SourceDynamics, restore
from src.world_model.m3w_source_continuation import atomic_checkpoint, learning_rate

VARIANTS = ('expected_cost', 'cost_supervised')


class CostDeferral(SourceDynamics):
    def __init__(self, width=480):
        super().__init__(width)
        self.risk_gate = nn.Linear(32, 1)
        nn.init.zeros_(self.risk_gate.weight)
        nn.init.zeros_(self.risk_gate.bias)

    def proposal_and_score(self, geometry, rgb, coverage, arm):
        n = len(geometry)
        if (arm not in ('mask_only', 'past_rgb') or geometry.shape != (n, self.width)
                or rgb.shape != (n, 8, 3, 32, 32) or coverage.shape != (n, 8, 1, 32, 32)):
            raise ValueError('Fixed observed-only schema required')
        rgb = torch.where(coverage > 0, rgb, torch.zeros_like(rgb))
        if arm == 'mask_only':
            rgb = torch.zeros_like(rgb)
        visual = self.image(torch.cat((rgb, coverage), 2).flatten(0, 1)).reshape(n, 8, 16)
        visual = visual * (coverage.sum((2, 3, 4)) > 0)[..., None]
        features = torch.cat((geometry, coverage.mean((2, 3, 4)), visual.flatten(1)), 1)
        hidden = self.head[:-1](features)
        raw = self.head[-1](hidden).reshape(n, 12, 2)
        proposal = raw/(1+torch.linalg.vector_norm(raw, dim=-1, keepdim=True))
        return proposal, self.risk_gate(hidden).squeeze(-1)

    def forward(self, geometry, rgb, coverage, arm):
        return self.proposal_and_score(geometry, rgb, coverage, arm)[0]


def hard_action(proposal, score):
    if proposal.ndim != 3 or proposal.shape[1:] != (12, 2) or score.shape != (len(proposal),):
        raise ValueError('Aligned proposal and one observed-input score per query required')
    return torch.where((score > 0)[:, None, None], proposal, torch.zeros_like(proposal))


def deferral_loss(proposal, score, target, scale, variant):
    if (variant not in VARIANTS+('dense_control',) or scale <= 0 or not np.isfinite(scale)
            or proposal.shape != target.shape or proposal.ndim != 3 or proposal.shape[1:] != (12, 2)
            or score.shape != (len(proposal),)):
        raise ValueError('Registered loss and aligned training labels required')
    error = torch.linalg.vector_norm(proposal-target, dim=-1).mean(1)/scale
    baseline = torch.linalg.vector_norm(target, dim=-1).mean(1)/scale
    p = score.sigmoid()
    expected = ((1-p)*baseline + p*error).mean()
    relative_gain = (baseline-error).detach()
    cost_fit = nn.functional.smooth_l1_loss(score, relative_gain)
    if variant == 'dense_control':
        loss = error.mean()
    elif variant == 'expected_cost':
        loss = expected
    else:
        loss = expected + cost_fit + .1*error.mean()
    return loss, dict(proposal_ade=error.mean(), baseline_ade=baseline.mean(),
        expected_action_risk=expected, cost_fit=cost_fit, mean_soft_gate=p.mean(),
        hard_rate=(score > 0).float().mean())


def extend_parent(model, parent, optimizer):
    missing, unexpected = model.load_state_dict(parent['model'], strict=False)
    if missing != ['risk_gate.weight', 'risk_gate.bias'] or unexpected:
        raise ValueError('Parent proposal architecture differs')
    nn.init.zeros_(model.risk_gate.weight)
    nn.init.zeros_(model.risk_gate.bias)
    old = copy.deepcopy(parent['optimizer'])
    if len(old['param_groups']) != 1:
        raise ValueError('One registered AdamW group required')
    previous = old['param_groups'][0]['params']
    current = optimizer.state_dict()['param_groups'][0]['params']
    if current[:-2] != previous or len(current) != len(previous)+2:
        raise ValueError('Only two new gate parameter tensors may be appended')
    old['param_groups'][0]['params'] = current
    optimizer.load_state_dict(old)


def fit_deferral(model, inputs, targets, ids, weights, *, parent, variant, config,
                 identity, checkpoint, heartbeat, snapshot, stop_at=None):
    ids, weights = np.asarray(ids, dtype=int), torch.as_tensor(weights, dtype=torch.float64)
    if (variant not in VARIANTS+('dense_control',) or parent['arm'] != 'mask_only'
            or parent['objective'] != 'ade' or parent['step'] != config['start_step']
            or len(np.unique(ids)) != len(ids) or not len(ids) or weights.shape != (len(ids),)
            or torch.any(weights <= 0) or not torch.isclose(weights.sum(), torch.tensor(1., dtype=weights.dtype))):
        raise ValueError('Verified mask parent and complete training-only complement required')
    np.testing.assert_array_equal(parent['train_ids'], ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator()
    scale = parent['normalizer']
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity or state['variant'] != variant or state['config'] != config or state['normalizer'] != scale:
            raise ValueError('Deferral resume identity changed')
        np.testing.assert_array_equal(state['train_ids'], ids)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        step, seconds, trace = state['step'], state['continuation_seconds'], state['trace']
        clipped, maximum = state['clipped_updates'], state['maximum_gradient_norm']
    else:
        extend_parent(model, parent, optimizer)
        state = parent
        step, seconds, trace, clipped, maximum = config['start_step'], 0., [], 0, 0.
    generator.set_state(state['sampler_rng'])
    torch.set_rng_state(state['torch_rng'])
    draws = state['draw_counts'].copy()
    first, started = step, time.monotonic()
    limit = config['updates'] if stop_at is None else min(stop_at, config['updates'])
    if limit < step or step > config['updates']:
        raise ValueError('Nondecreasing fixed update budget required')

    def payload():
        return dict(identity=identity, variant=variant, config=config, normalizer=scale,
            model=model.state_dict(), optimizer=optimizer.state_dict(), train_ids=ids,
            sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(), draw_counts=draws,
            step=step, continuation_seconds=seconds+time.monotonic()-started,
            trace=trace, clipped_updates=clipped, maximum_gradient_norm=maximum)

    if not checkpoint.exists():
        atomic_checkpoint(checkpoint, payload())
    if step in config['milestones']:
        snapshot(torch.load(checkpoint, map_location='cpu', weights_only=False))
    model.train()
    try:
        while step < limit:
            rate = learning_rate('cosine', step, config)
            for group in optimizer.param_groups:
                group['lr'] = rate
            local = torch.multinomial(weights, config['batch_size'], replacement=True, generator=generator).numpy()
            index = ids[local]
            features, frame = inputs(index)
            proposal, score = model.proposal_and_score(*features, 'mask_only')
            proposal = restore(proposal, *frame)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = deferral_loss(proposal, score, targets(index), scale, variant)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite deferral loss')
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
            optimizer.step()
            step += 1
            np.add.at(draws, local, 1)
            clipped += int(grad > 5)
            maximum = max(maximum, grad)
            if step == config['start_step']+1 or step % 100 == 0:
                trace.append(dict(step=step, learning_rate=rate, loss=float(loss.detach()), gradient_norm=grad,
                                  **{k:float(v.detach()) for k,v in terms.items()}))
            if step == limit or step % config['checkpoint_every'] == 0 or step in config['milestones']:
                state = payload()
                atomic_checkpoint(checkpoint, state)
                heartbeat(state='training', step=step, continuation_seconds=state['continuation_seconds'],
                          loss=float(loss.detach()), soft_gate=float(terms['mean_soft_gate'].detach()))
                if step in config['milestones']:
                    snapshot(state)
                    model.train()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_completed_step=step)
        raise
    model.eval()
    return dict(complete=step == config['updates'], step=step, new_updates=step-first,
        additional_updates=step-config['start_step'], continuation_seconds=seconds+time.monotonic()-started,
        clipped_updates=clipped, maximum_gradient_norm=maximum, trace=trace,
        total_draws=int(draws.sum()), mean_draws_per_row=float(draws.mean()))


def predict_deferral(model, inputs, ids):
    model.eval()
    proposals, scores = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 128):
            features, frame = inputs(np.asarray(ids[start:start+128]))
            proposal, score = model.proposal_and_score(*features, 'mask_only')
            proposals.append(restore(proposal, *frame).numpy())
            scores.append(score.numpy())
    return np.concatenate(proposals), np.concatenate(scores)
