"""Past-conditioned motion interpolation with missing-label disagreement costs."""
import torch
from torch import nn


class CompletionMotionBlend(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(12, 24), nn.SiLU(), nn.Linear(24, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, reference, proposal):
        speed = torch.linalg.vector_norm(history.diff(dim=1), dim=2)
        scale = speed.square().mean(1).sqrt().clamp_min(.05)
        distance = torch.linalg.vector_norm(proposal-reference, dim=2)
        length = 12.*scale
        features = torch.cat((speed/scale[:, None], torch.log1p(scale)[:, None],
            torch.stack((distance.mean(1)/length, distance.amax(1)/length,
                         torch.linalg.vector_norm(reference[:, -1], dim=1)/length,
                         torch.linalg.vector_norm(proposal[:, -1], dim=1)/length), dim=1)), dim=1)
        return self.head(features.clamp(-16., 16.)).sigmoid()


def objectives(pred, reference, target, valid, scale, left, right):
    known_error = torch.linalg.vector_norm(torch.where(valid[..., None], pred-target, 0.), dim=2)
    individual = (known_error.sum(1)/valid.sum(1).clamp_min(1)/scale).mean()
    unknown_distance = torch.linalg.vector_norm(pred-reference, dim=2).amax(1)
    individual_debit = (torch.where(valid.any(1), 0., unknown_distance)/scale).mean()
    observed = valid[left] & valid[right]
    relative = pred[left]-pred[right]
    rel_target = target[left]-target[right]
    rel_reference = reference[left]-reference[right]
    error = torch.linalg.vector_norm(torch.where(observed[..., None], relative-rel_target, 0.), dim=2)
    debit = torch.linalg.vector_norm(torch.where(observed[..., None], 0., relative-rel_reference), dim=2)
    pair_scale = (scale[left]*scale[right]).sqrt()
    pair = (error.mean(1)/pair_scale).mean()
    pair_debit = (debit.mean(1)/pair_scale).mean()
    return individual, pair, individual_debit, pair_debit
