"""Learn relative scores over two plans sharing a realized edit budget."""

import torch
from torch.nn.functional import softplus


def plan_values(logits, choices, group, scale):
    """Sum increments from no-edit, then normalize both plans identically."""
    values = logits.gather(1, choices) - logits[:, :1]
    return values.new_zeros((len(scale), 2)).index_add(0, group, values) / scale[:, None]


def objective(logits, choices, group, scale, *, pairwise):
    good, bad = plan_values(logits, choices, group, scale).unbind(1)
    if pairwise:
        return softplus(bad - good).mean()
    return ((softplus(-good) + softplus(bad)) * .5).mean()
