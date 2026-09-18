"""Frozen-image temporal readout; supplied past annotations, not sensor-as-of."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

import numpy as np
import torch
from torch import nn

ARMS = ('geometry', 'current', 'sequence')


def normalized_embedding(features, coverage):
    if (features.ndim != 2 or features.shape[1] != 512
            or coverage.shape != (len(features),) or not torch.isfinite(features).all()
            or not torch.isfinite(coverage).all() or torch.any((coverage < 0) | (coverage > 1))):
        raise ValueError('Finite 512-dimensional embeddings and valid coverage required')
    result = nn.functional.normalize(features, dim=-1)
    return torch.where(coverage[:, None] > 0, result, torch.zeros_like(result))


class TemporalSourceDynamics(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Registered visual arm required')
        self.arm = arm
        self.geometry = nn.Sequential(nn.Linear(480, 64), nn.SiLU())
        self.visual = nn.Sequential(nn.Linear(512, 32), nn.SiLU())
        self.temporal = nn.GRU(33, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64+32+32+8, 64), nn.SiLU(), nn.Linear(64, 24))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, geometry, appearance, coverage, engine_tag='mask_only'):
        # Frozen legacy optimizer supplies this tag; self.arm owns the ablation.
        n = len(geometry)
        if (engine_tag != 'mask_only' or geometry.shape != (n, 480)
                or appearance.shape != (n, 8, 512) or coverage.shape != (n, 8)):
            raise ValueError('Past-only geometry, eight visual tokens and coverage required')
        if self.arm == 'geometry':
            appearance = torch.zeros_like(appearance)
        elif self.arm == 'current':
            appearance = appearance[:, -1:, :].expand(-1, 8, -1)
        visual = self.visual(appearance)
        visual = torch.where(coverage[..., None] > 0, visual, torch.zeros_like(visual))
        sequence, _ = self.temporal(torch.cat((visual, coverage[..., None]), dim=-1))
        raw = self.head(torch.cat((self.geometry(geometry), sequence[:, -1],
                                  sequence[:, -1]-sequence[:, 0], coverage), dim=-1)).reshape(n, 12, 2)
        return raw/(1+torch.linalg.vector_norm(raw, dim=-1, keepdim=True))


def embedding_lookup(query_ids, admitted_ids, query_embedding_rows):
    ids = np.asarray(query_ids, dtype=int)
    admitted_ids = np.asarray(admitted_ids, dtype=int)
    if (ids.ndim != 1 or not len(admitted_ids) or np.any(np.diff(admitted_ids) <= 0)
            or query_embedding_rows.shape != (len(admitted_ids), 8)):
        raise ValueError('Sorted unique admitted IDs and eight-step row alignment required')
    loc = np.searchsorted(admitted_ids, ids)
    if np.any(loc >= len(admitted_ids)) or not np.array_equal(admitted_ids[loc], ids):
        raise ValueError('Outer/main/unknown query prohibited')
    return query_embedding_rows[loc]
