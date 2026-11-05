"""Whole-fitting all/easy loss attribution, not a generalization certificate."""
import copy
import numpy as np
import torch
from src.world_model import m3w_kinematic_risk as kin
from src.world_model.m3w_frozen_readout_refit import query_weights

VARIANTS = ('joint', 'easy_only', 'easy_protected')


def batches(causal, maximum=1024):
    pending, size = [], 0
    for group in causal['groups']:
        if pending and size+len(group) > maximum:
            yield np.concatenate(pending), np.repeat(np.arange(len(pending)), [len(g) for g in pending])
            pending, size = [], 0
        pending.append(group); size += len(group)
    if pending:
        yield np.concatenate(pending), np.repeat(np.arange(len(pending)), [len(g) for g in pending])


def task_losses(prediction, target, allowed, weights):
    if prediction.shape != target.shape or prediction.shape[-1] != 2 or target.requires_grad:
        raise ValueError('Detached aligned all/easy supervision required')
    row = (((prediction-target).square()*allowed[..., None]).sum(1)
        /allowed.sum(1).clamp_min(1)[:, None])
    return (row.double()*weights[:, None]).sum(0)


def full(model, data, counts, gradient=False, guard=lambda: None):
    c = data['causal']; z = kin.batch_features(data, 'kinematic')
    weights = query_weights(c, counts)
    target = kin.parent.signed(data['truth_by_arm']['bound_aware'])/torch.from_numpy(kin.parent.normalizer(data['preprocess']))
    params = list(model.parameters()); total = np.zeros(2, float)
    gradients = [[torch.zeros_like(p, dtype=torch.float64) for p in params] for _ in range(2)] if gradient else None
    errors = np.zeros(2, float); false_safe = np.zeros(2, np.int64)
    for ix, seg in batches(c):
        guard()
        with torch.set_grad_enabled(gradient):
            pred = model(z[ix], torch.from_numpy(seg), int(seg.max())+1)
            terms = task_losses(pred, target[ix], torch.from_numpy(c['allowed'][ix]), torch.from_numpy(weights[ix]))
        total += terms.detach().numpy()
        if gradient:
            for channel in range(2):
                gs = torch.autograd.grad(terms[channel], params, retain_graph=channel == 0)
                for out, g in zip(gradients[channel], gs):
                    out.add_(g.detach().double())
        y, p = target[ix].numpy(), pred.detach().numpy()
        known = data['observed_future'][ix, None, None] & c['allowed'][ix, :, None]
        wrong = ((p <= 0) & (y > 0)) | ((p > 0) & (y < 0))
        errors += (weights[ix, None, None]*abs(y)*wrong*known).sum((0, 1))
        false_safe += (known & (p <= 0) & (y > 0)).sum((0, 1))
    return dict(task_MSE=total.tolist(), observed_sign_regret=errors.tolist(),
        false_safe_channels=false_safe.tolist()), gradients


def geometry(a, e):
    na, ne = float(a.norm()), float(e.norm()); dot = float(a@e)
    joint = (a+e)/2; nj = float(joint.norm())
    return dict(all_norm=na, easy_norm=ne, all_easy_dot=dot,
        all_easy_cosine=dot/(na*ne) if na > 0 and ne > 0 else None,
        joint_easy_dot=float(joint@e),
        joint_easy_cosine=float(joint@e)/(nj*ne) if nj > 0 and ne > 0 else None,
        joint_first_order_easy_harm=bool(joint@e < 0) if ne > 0 else None)


def combine(model, gradients, variant):
    if variant not in VARIANTS:
        raise ValueError('Fixed diagnostic variants required')
    a, e = gradients; names = [name for name, _ in model.named_parameters()]
    shared = [i for i, name in enumerate(names) if not name.startswith('excess.')]
    ga = torch.cat([a[i].reshape(-1) for i in shared]); ge = torch.cat([e[i].reshape(-1) for i in shared])
    dot, square = float(ga@ge), float(ge@ge)
    adjusted = [g.clone() for g in a]
    if variant == 'easy_protected' and dot < 0 and square > 0:
        for i in shared:
            adjusted[i] -= dot/square*e[i]
    combined = [y/2 if variant == 'easy_only' else (x+y)/2 for x, y in zip(adjusted, e)]
    return combined, geometry(ga, ge)


def virtual(model, state, gradients, variant):
    out = copy.deepcopy(model)
    opt = torch.optim.AdamW(out.parameters(), lr=state['settings']['learning_rate'], weight_decay=.0001)
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    gs, relation = combine(out, gradients, variant)
    for p, g in zip(out.parameters(), gs):
        p.grad = g.to(p.dtype).clone()
    norm = torch.nn.utils.clip_grad_norm_(out.parameters(), state['settings']['gradient_clip'], error_if_nonfinite=True)
    opt.step()
    if not all(torch.isfinite(p).all() for p in out.parameters()):
        raise FloatingPointError('Nonfinite disposable optimizer update')
    return out, dict(shared_relation=relation, gradient_norm_before_clip=float(norm))
