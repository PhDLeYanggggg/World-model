from __future__ import annotations

import torch


def project_speed_acceleration(state: torch.Tensor, max_speed: float = 50.0, max_accel: float = 20.0) -> torch.Tensor:
    out = state.clone()
    v = out[..., 2:4]
    speed = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
    out[..., 2:4] = v * torch.clamp(max_speed / speed, max=1.0)
    if out.shape[-1] >= 6:
        a = out[..., 4:6]
        accel = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-6)
        out[..., 4:6] = a * torch.clamp(max_accel / accel, max=1.0)
    return out
