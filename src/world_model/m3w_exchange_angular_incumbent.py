"""Train the angular head against the incumbent it would replace."""

import torch


ARMS = ('dual_reference', 'incumbent_mean', 'incumbent_tail')


def gain(pred, ref, target, valid, scale):
    error = torch.linalg.vector_norm(pred-target, dim=-1)
    base = torch.linalg.vector_norm(ref-target, dim=-1)
    debit = torch.linalg.vector_norm(pred-ref, dim=-1)
    return ((base-error)*valid-debit*(~valid)).sum(1)/(pred.shape[1]*scale)


def objective(pred, cv, ols, anchor, target, valid, scale, group_size, arm):
    if arm not in ARMS or len(pred) % group_size:
        raise ValueError('Require registered, equally sized donor batches')
    relative = -torch.minimum(gain(pred, cv, target, valid, scale),
                              gain(pred, ols, target, valid, scale)).mean()
    if arm == 'dual_reference':
        return relative
    group = -gain(pred, anchor, target, valid, scale).reshape(-1, group_size).mean(1)
    if arm == 'incumbent_mean':
        return relative+group.mean()
    if len(group) < 2:
        raise ValueError('Tail objective needs at least two donor groups')
    return relative+.5*group.mean()+.5*group.topk(2).values.mean()
