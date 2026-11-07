"""Direct horizon-dependent velocity integration from past geometry."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import torch
from torch import nn


class KinematicForecaster(nn.Module):
    """Learn which measured velocities to trust, without a free-form residual."""

    def __init__(self, use_peers):
        super().__init__()
        self.use_peers = use_peers
        self.peer = nn.Sequential(nn.Linear(7, 24), nn.SiLU(), nn.Linear(24, 24))
        self.decode = nn.Sequential(nn.Linear(39, 48), nn.SiLU(), nn.Linear(48, 48),
                                    nn.SiLU(), nn.Linear(48, 12 * 8))

    def forward(self, inputs):
        velocity = inputs['history'][..., :2].diff(dim=1)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        scale = speed.square().mean(1).sqrt().clamp_min(.01)
        v = velocity / scale[:, None, None]
        mean = v.mean(1)
        features = torch.cat((speed / scale[:, None], (v * mean[:, None]).sum(-1),
                              torch.log1p(scale)[:, None]), dim=1)
        context = velocity.new_zeros((len(velocity), 24))
        if self.use_peers:
            valid = inputs['neighbor_mask'].all(-1)
            history = torch.where(valid[..., None, None], inputs['neighbors'][..., :2], 0.)
            pos = history[:, :, -1] - inputs['history'][:, None, -1, :2]
            pos = pos / scale[:, None, None]
            peer_v = (history[:, :, -1] - history[:, :, -4]) / (3 * scale[:, None, None])
            delta = peer_v - mean[:, None]
            distance = torch.linalg.vector_norm(pos, dim=-1)
            toward = (delta * pos).sum(-1) / distance.clamp_min(.01)
            pair = torch.stack((torch.log1p(distance), toward,
                                torch.linalg.vector_norm(peer_v, dim=-1),
                                torch.linalg.vector_norm(delta, dim=-1),
                                (peer_v * mean[:, None]).sum(-1),
                                (pos * mean[:, None]).sum(-1) / distance.clamp_min(.01),
                                valid.sum(1, keepdim=True).expand_as(distance) / 8), dim=-1)
            encoded = self.peer(pair.clamp(-16., 16.))
            context = torch.where(valid[..., None], encoded, 0.).sum(1)
            context = context / valid.sum(1, keepdim=True).clamp_min(1)
        params = self.decode(torch.cat((features.clamp(-16., 16.), context), 1)).reshape(-1, 12, 8)
        mixture = params[..., :7].softmax(-1)
        gain = 2 * params[..., 7:].sigmoid()
        return (gain * (mixture @ velocity)).cumsum(1)
