"""Deterministic, focal-graph adaptation of the pinned Social-STGCNN core."""

from functools import lru_cache
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from torch import nn

from src.world_model.m3w_opendd_comoving_eqmotion import frame

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/stage_cvpr2027_experiments/social_stgcnn_author'
UPSTREAM = 'ebd57aaf34d84763825d05cf9d4eff738d8c96bb'
DIGESTS = {
    'model.py': '78a2e005c6f8ae8b89af5d59319f08a3a98e127c296ea9f585c2b2661c1f3f05',
    'LICENSE': '364f9d362d639560c45bcc7450e4f39ec80474923f116911c8745c28f4d56985',
}
ARMS = ('social_peers', 'social_ego')
ARCH = dict(n_stgcnn=1, n_txpcnn=5, input_feat=2, output_feat=2, seq_len=8, pred_seq_len=12, kernel_size=3)


@lru_cache(maxsize=1)
def author():
    for name, digest in DIGESTS.items():
        if sha256((SOURCE/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Social-STGCNN source or license does not match the pinned release')
    spec = spec_from_file_location('_m3w_social_stgcnn_author', SOURCE/'model.py')
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BatchedGraphConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.conv = original.conv
        self.kernel_size = original.kernel_size

    def forward(self, x, adjacency):
        z = self.conv(x)
        n, kc, t, v = z.shape
        z = z.reshape(n, self.kernel_size, kc//self.kernel_size, t, v)
        return torch.einsum('nkctv,nkvw->nctw', z, adjacency).contiguous(), adjacency


def graph(motion, active):
    """Author inverse-increment affinity and normalized Laplacian, per sample."""
    if motion.ndim != 4 or active.shape != motion.shape[:2] or active.dtype != torch.bool:
        raise ValueError('Invalid masked observation graph')
    x = motion.transpose(1, 2)
    distance = (x[..., :, None, :]-x[..., None, :, :]).norm(dim=-1)
    pairs = (active[:, None, :, None] & active[:, None, None, :])
    # Treat equal increments as no edge, following the author kernel.
    edges = torch.where(pairs & (distance > 0.), distance.clamp_min(1e-12).reciprocal(), 0.)
    diagonal = torch.diag_embed(active.to(x.dtype))[:, None]
    a = edges+diagonal
    scale = a.sum(dim=-1).clamp_min(1.).rsqrt()
    return diagonal-a*scale[..., None]*scale[..., None, :]


class Forecast(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown Social-STGCNN arm')
        self.arm = arm
        self.net = author().social_stgcnn(**ARCH)
        for layer in self.net.st_gcns:
            layer.gcn = BatchedGraphConv(layer.gcn)

    def forward(self, ego, peers, mask):
        mode = 'comoving_peers' if self.arm == 'social_peers' else 'comoving_ego'
        z, scale, restore = frame(ego, peers, mask, mode)
        history = torch.cat((z['history'][:, None, :, :2], z['neighbors'][..., :2]), dim=1)
        active = torch.cat((torch.ones_like(mask[:, :1]), z['neighbor_mask'][..., 0]), dim=1)
        if self.arm == 'social_ego':
            history, active = history[:, :1], active[:, :1]
        delta = torch.diff(history, dim=2, prepend=history[:, :, :1])
        x = delta.permute(0, 3, 2, 1).contiguous()
        out, _ = self.net(x, graph(delta, active))
        return out[..., 0].transpose(1, 2).cumsum(1)*scale[:, None, None]+restore
