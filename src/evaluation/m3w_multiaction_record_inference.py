"""Inference-only shape repair; frozen training/model code stays unchanged."""
import numpy as np
import torch
from src.world_model import m3w_multiaction_record_policy as trained


def predict(state, features, anchor, allowed, *, arm, chunk_size=8192):
    if arm != state['arm']:
        raise ValueError('Frozen action bank required')
    features, anchor, allowed = map(np.asarray, (features, anchor, allowed))
    if (features.shape != (len(anchor), 7, 12) or anchor.shape != allowed.shape
            or anchor.shape[1] != 7 or anchor.dtype != bool or allowed.dtype != bool
            or not np.isfinite(features).all()):
        raise ValueError('Finite causal features and explicit action axes required')
    model = trained.RecordActionPolicy(state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    # A one-element Torch tensor becomes a scalar NumPy index; keep NumPy axes explicit.
    indices = np.asarray(trained.ARMS[arm], dtype=np.int64)
    action_ids = torch.from_numpy(indices)
    choices, ranks = [], []
    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            end = start+chunk_size
            f = torch.from_numpy(np.clip(features[start:end][:, indices], -8, 8).astype(np.float32))
            a = torch.from_numpy(anchor[start:end][:, indices])
            ok = torch.from_numpy(allowed[start:end][:, indices])
            logits = model(f, a, action_ids); _, selected = trained.categorical(logits, ok, .1)
            local = selected.numpy(); active = local >= 0
            chosen = np.full(len(local), -1, np.int64); chosen[active] = indices[local[active]]
            rank = np.zeros(len(local)); rank[active] = logits.numpy()[active, local[active]]
            choices.append(chosen); ranks.append(rank)
    return np.concatenate(choices), np.concatenate(ranks)
