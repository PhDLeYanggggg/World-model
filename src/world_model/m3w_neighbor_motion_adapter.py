"""Matched past-only context adapters over a frozen ego-motion forecast."""
import torch
from torch import nn
from torch.nn.functional import interpolate

ARMS = ('ego', 'pooled_motion', 'local_relative_motion')


class NeighborMotionAdapter(nn.Module):
    def __init__(self, backbone, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown context arm')
        self.arm = arm
        self.backbone = backbone.requires_grad_(False)
        self.peer = nn.Sequential(nn.Linear(7, 24), nn.SiLU(), nn.Linear(24, 12))
        self.decode = nn.Sequential(nn.Linear(31, 48), nn.SiLU(), nn.Linear(48, 6))

    def forward(self, x):
        h = x['history'][..., :2]
        d = h.diff(dim=1)
        v = (d[:, -3:]*d.new_tensor([.3, .4, .3])[None, :, None]).sum(1)
        speed = d.square().sum(2)
        scale = speed.mean(1).sqrt().clamp_min(.05)
        largest = d[torch.arange(len(h), device=h.device), speed.argmax(1)]
        axis = torch.where((v.square().sum(1) > 1e-10)[:, None], v, largest)
        axis = axis/torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
        frame = torch.stack((axis, torch.stack((-axis[:, 1], axis[:, 0]), 1)), 2)
        local = (d @ frame)/scale[:, None, None]
        pooled, interaction = h.new_zeros((len(h), 3)), h.new_zeros((len(h), 12))
        if self.arm != 'ego':
            valid = x['neighbor_mask'].all(2)
            nh = torch.where(valid[..., None, None], x['neighbors'][..., :2], 0.)
            n = valid.sum(1).clamp_min(1)
            peer_v = (nh[:, :, -1]-nh[:, :, -4])/3.
            mean_v = torch.where(valid[..., None], peer_v, 0.).sum(1)/n[:, None]
            pooled = torch.cat(((mean_v[:, None] @ frame).squeeze(1)/scale[:, None],
                                valid.sum(1, keepdim=True)/8.), 1)
            if self.arm == 'local_relative_motion':
                position = ((nh[:, :, -1]-h[:, None, -1]) @ frame)/scale[:, None, None]
                relative = ((peer_v-v[:, None]) @ frame)/scale[:, None, None]
                centered = ((peer_v-mean_v[:, None]) @ frame)/scale[:, None, None]
                dist = torch.linalg.vector_norm(position, dim=2)
                closing = -(position*relative).sum(2)/dist.clamp_min(.01)
                values = torch.cat((position/dist[..., None].clamp_min(.01), relative, centered,
                                    closing[..., None]), 2)
                weight = valid/(1.+dist)
                interaction = (self.peer(values.clamp(-16., 16.))*weight[..., None]).sum(1)
                interaction = interaction/weight.sum(1, keepdim=True).clamp_min(1e-6)
        features = torch.cat((local.flatten(1), torch.log1p(scale)[:, None],
                              (torch.linalg.vector_norm(v, dim=1)/scale)[:, None],
                              pooled, interaction), 1)
        knots = self.decode(features).reshape(-1, 2, 3).tanh()
        inc = interpolate(knots, size=12, mode='linear', align_corners=True).transpose(1, 2)
        correction = ((inc @ frame.transpose(1, 2))*scale[:, None, None]).cumsum(1)
        return self.backbone(x)+correction
