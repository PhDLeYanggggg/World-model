"""One fixed, detached norm-parity repair for the existing admission auxiliary."""
from contextlib import contextmanager
import copy
from pathlib import Path

import torch
from src.world_model import m3w_admission_cost as frozen
from src.world_model.m3w_admission_monitor import vector

RULE = dict(target_ratio=1., max_scale=1e6, zero_tolerance=1e-12)


def balanced_total(anchor, auxiliary, parameters):
    ga, gb = vector(anchor, parameters).double(), vector(auxiliary, parameters).double()
    if not torch.isfinite(ga).all() or not torch.isfinite(gb).all():
        raise FloatingPointError('Finite component gradients required')
    an, bn = float(ga.norm()), float(gb.norm())
    raw = RULE['target_ratio']*an/bn if an > 0 and bn > RULE['zero_tolerance'] else 0.
    scale = min(raw, RULE['max_scale'])
    return anchor + scale*auxiliary, dict(scale=scale, anchor_gradient_norm=an,
        auxiliary_gradient_norm=bn, capped=raw > RULE['max_scale'],
        weighted_auxiliary_over_anchor=scale*bn/an if an else None)


@contextmanager
def installed(settings, saved=None):
    active = frozen.parent.active
    base, detector = active.completion.base, active.detector
    old_ctor, old_save = detector.DetectorCostHead, base.save_checkpoint
    state = (copy.deepcopy(saved) if saved is not None else dict(rule=RULE.copy(), step=0,
        zero_scale_steps=0, capped_steps=0, scale_sum=0., scale_max=0., rows=[]))
    if state['rule'] != RULE:
        raise ValueError('Frozen balancing rule changed')
    model = None
    with frozen.parent.installed('upper_score'):
        old_loss, old_name = base.continuous_loss, active.completion.OBJECTIVE
        def ctor(*args, **kwargs):
            nonlocal model
            model = old_ctor(*args, **kwargs)
            return model
        def loss(prediction, target, real, rms, **kwargs):
            terms = old_loss(prediction, target, real, rms, **kwargs)
            auxiliary, info = frozen.admission_penalty(prediction, target, real, rms)
            total, stats = balanced_total(terms['total'], auxiliary, list(model.parameters()))
            state['step'] += 1
            state['zero_scale_steps'] += int(stats['scale'] == 0)
            state['capped_steps'] += int(stats['capped'])
            state['scale_sum'] += stats['scale']
            state['scale_max'] = max(state['scale_max'], stats['scale'])
            if state['step'] == 1 or state['step'] % settings['heartbeat_every'] == 0 or state['step'] == settings['steps']:
                state['rows'].append(dict(step=state['step'], **stats, **info,
                    anchor_loss=float(terms['total'].detach()), auxiliary_loss=float(auxiliary.detach()),
                    balanced_total_loss=float(total.detach())))
            return dict(terms, total=total)
        def save(path, checkpoint):
            if checkpoint['step'] != state['step']:
                raise ValueError('Balance telemetry and atomic checkpoint must align')
            old_save(path, dict(checkpoint, gradient_balance=copy.deepcopy(state)))
        detector.DetectorCostHead = ctor
        base.continuous_loss, base.save_checkpoint = loss, save
        active.completion.OBJECTIVE = old_name + '+detached_equal_gradient_admission_target1_cap1e6'
        try:
            yield state
        finally:
            detector.DetectorCostHead = old_ctor
            base.continuous_loss, base.save_checkpoint = old_loss, old_save
            active.completion.OBJECTIVE = old_name


def fit(data, **kwargs):
    path = Path(kwargs['path'])
    saved = None
    if path.exists() and kwargs.get('resume'):
        checkpoint = frozen.parent.active.completion.base.read_checkpoint(path)
        saved = checkpoint['gradient_balance']
        if saved['step'] != checkpoint['step']:
            raise ValueError('Resume balance statistics and model step differ')
    with installed(kwargs['settings'], saved):
        return frozen.parent.active.fit(data, **kwargs)
