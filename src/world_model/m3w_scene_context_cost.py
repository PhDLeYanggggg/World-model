"""One-factor static scene-context cost ablation; trajectory producer unchanged."""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_multiaction_cost as base

OriginalCostHead = base.ActionCostHead
SCENE_DIMENSION = 18


class SceneCausalFeatures:
    def __init__(self, motion, scene):
        if scene.shape != (*motion.shape[:2], SCENE_DIMENSION) or not torch.isfinite(scene).all():
            raise ValueError('18 aligned finite static/past/rollout features required')
        self.motion, self.scene = motion, scene
        self.shape = (*motion.shape[:2], motion.shape[-1]+SCENE_DIMENSION)

    def __getitem__(self, rows):
        motion, scene = self.motion[rows], self.scene[rows]
        if motion.ndim != 3:
            raise ValueError('Explicit row batches preserve candidate axes')
        return torch.cat((motion, scene), -1)


class SceneCostHead(OriginalCostHead):
    def __init__(self, dimension, width, seed, bias, *, enabled=True):
        if dimension <= SCENE_DIMENSION:
            raise ValueError('Motion plus18static scene fields required')
        self.motion_dimension = dimension-SCENE_DIMENSION
        self.enabled = enabled
        super().__init__(self.motion_dimension, width, seed, bias)
        # Zero initialization preserves the existing predictor and random stream.
        self.scene_weight = nn.Parameter(torch.zeros(width, SCENE_DIMENSION))

    def forward(self, x, envelope, action_ids):
        motion = x[..., :self.motion_dimension]
        if not self.enabled:
            return super().forward(motion, envelope, action_ids)
        if x.shape[-1] != self.motion_dimension+SCENE_DIMENSION or envelope.shape != x.shape[:2]:
            raise ValueError('Aligned causal scene fields and envelope required')
        code = F.one_hot(action_ids, 7).to(x.dtype).expand(len(x), -1, -1)
        hidden = self.network[0](torch.cat((motion, code), -1))
        hidden = hidden+F.linear(x[..., self.motion_dimension:], self.scene_weight)
        raw = self.network[2](self.network[1](hidden))
        fractions = torch.softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
        benefit, harm = (envelope[..., None]*fractions[..., :2]).unbind(-1)
        reference = F.softplus(raw[..., 2]); benefit = torch.minimum(benefit, reference)
        return torch.stack((benefit, harm, reference, reference*torch.sigmoid(raw[..., 3]),
                            harm*torch.sigmoid(raw[..., 4])), -1)


@contextmanager
def installed(enabled=True):
    original = base.ActionCostHead
    base.ActionCostHead = lambda *a, **k:SceneCostHead(*a, **k, enabled=enabled)
    try:
        yield
    finally:
        base.ActionCostHead = original


def preprocessing(motion, scene, envelope, labels, scales, sites):
    extra = base.preprocessing(scene, envelope, labels, scales, sites)
    return dict(motion, mean=np.r_[motion['mean'], extra['mean']],
                std=np.r_[motion['std'], extra['std']])


def predict(state, x, scene, envelope, scales, *, enabled=True):
    if scene.shape != (*x.shape[:2], SCENE_DIMENSION) or not np.isfinite(scene).all():
        raise ValueError('Finite causal scene features required')
    with installed(enabled):
        return base.predict(state, np.concatenate((x, scene), -1), envelope, scales)


def assert_neutral_state(reference, padded):
    """Disabled branch must reproduce real old weights,Adam moments and draws."""
    for key in ('draw_hash', 'torch_rng', 'sampler_rng', 'row_draws', 'trace', 'step', 'settings'):
        base.exact(reference[key], padded[key])
    for key in reference['model']:
        base.exact(reference['model'][key], padded['model'][key])
        base.exact(reference['initial_model'][key], padded['initial_model'][key])
    if torch.count_nonzero(padded['model']['scene_weight']):
        raise ValueError('Disabled scene projection changed')
    # Top-level scene_weight precedes the four existing network parameters.
    old, new = reference['optimizer'], padded['optimizer']
    if set(new['state']) != {i+1 for i in old['state']}:
        raise ValueError('Unexpected disabled optimizer state')
    for i, value in old['state'].items():
        base.exact(value, new['state'][i+1])
    for key, value in reference['preprocess'].items():
        other = padded['preprocess'][key]
        base.exact(value, other[:len(value)] if key in ('mean', 'std') else other)
