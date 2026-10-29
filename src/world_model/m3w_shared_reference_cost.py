"""Parameter-matched cost head with a single causal reference token per query."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from src.world_model import m3w_multiaction_cost as base

OriginalCostHead = base.ActionCostHead


class PairedCausalFeatures:
    """Keep one floor token per row; concatenate only the sampled batch."""
    def __init__(self, candidate, floor):
        if (candidate.ndim != 3 or floor.shape != (len(candidate), candidate.shape[-1])
                or not torch.isfinite(floor).all()):
            raise ValueError('One finite causal floor token per candidate row required')
        self.candidate, self.floor = candidate, floor
        self.shape = (*candidate.shape[:-1], 2*candidate.shape[-1])

    def __getitem__(self, rows):
        x = self.candidate[rows]; floor = self.floor[rows]
        if x.ndim != 3:
            raise ValueError('Explicit row batches preserve candidate axes')
        return torch.cat((x, floor[:, None].expand(-1, x.shape[1], -1)), -1)


class SharedReferenceCostHead(OriginalCostHead):
    def __init__(self, dimension, width, seed, bias):
        if dimension % 2:
            raise ValueError('Paired candidate/floor features have equal widths')
        self.causal_dimension = dimension//2
        super().__init__(self.causal_dimension, width, seed, bias)

    def forward(self, x, envelope, action_ids):
        if (x.ndim != 3 or x.shape[-1] != 2*self.causal_dimension
                or envelope.shape != x.shape[:2] or len(action_ids) != x.shape[1]):
            raise ValueError('Causal candidate/floor token pairs required')
        candidate = x[..., :self.causal_dimension]
        floor = x[:, 0, self.causal_dimension:]
        code = F.one_hot(action_ids, 7).to(x.dtype).expand(len(x), -1, -1)
        raw = self.network(torch.cat((candidate, code), -1))
        # Reference target belongs to the query/floor, not to the proposed action.
        shared = self.network(torch.cat((floor, torch.zeros((len(x), 7), dtype=x.dtype, device=x.device)), -1))
        fraction = torch.softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
        benefit, harm = (envelope[..., None]*fraction[..., :2]).unbind(-1)
        reference = F.softplus(shared[:, 2])[:, None].expand_as(harm)
        benefit = torch.minimum(benefit, reference)
        easy_reference = reference*torch.sigmoid(shared[:, 3])[:, None]
        easy_harm = harm*torch.sigmoid(raw[..., 4])
        return torch.stack((benefit, harm, reference, easy_reference, easy_harm), -1)


@contextmanager
def installed(enabled=True):
    original = base.ActionCostHead
    if enabled:
        base.ActionCostHead = SharedReferenceCostHead
    try:
        yield
    finally:
        base.ActionCostHead = original


def predict(state, x, floor, envelope, scales, *, chunk_size=1024):
    x, floor, envelope, scales = map(np.asarray, (x, floor, envelope, scales))
    base.validate_arrays(x, envelope, np.ones(envelope.shape, bool))
    if (floor.shape != (len(x), x.shape[-1]) or not np.isfinite(floor).all()
            or state['dimension'] != 2*x.shape[-1] or scales.shape != (len(x),)
            or not np.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError('Finite past-only paired features and frozen scales required')
    model = SharedReferenceCostHead(state['dimension'], state['settings']['width'],
        state['seed'], state['preprocess']['bias'])
    model.load_state_dict(state['model']); model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            end = start+chunk_size
            z = PairedCausalFeatures(torch.from_numpy(base.standardized(x[start:end], state['preprocess'])),
                torch.from_numpy(base.standardized(floor[start:end], state['preprocess'])))
            env = torch.from_numpy((envelope[start:end]/scales[start:end, None]).astype(np.float32))
            p = model(z[np.arange(len(z.floor))], env, torch.arange(7)).numpy()*scales[start:end, None, None]
            if not all(np.array_equal(p[..., k], np.repeat(p[:, :1, k], 7, axis=1)) for k in (2, 3)):
                raise ValueError('Shared reference invariance failed')
            output.append(p.astype(float))
    return np.concatenate(output)
