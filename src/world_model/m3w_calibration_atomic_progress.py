"""One atomic grid-prefix checkpoint; model/risk/allocation semantics unchanged."""
from pathlib import Path
import numpy as np
from src.world_model.m3w_causal_descriptor_head import read_checkpoint, save_checkpoint


def validate(state, identity, rows):
    if state['identity'] != identity:
        raise ValueError('Same frozen policy/inference contract required')
    choices = state['choices']; n = len(state['allocation'])
    if (choices.shape != (13, 4, rows) or choices.dtype != np.int8
            or not 0 <= n <= 13 or (choices < -1).any() or (choices >= 7).any()
            or np.any(choices[n:] != -1)):
        raise ValueError('Committed grid prefix and unchanged int8 choice schema required')
    return state


def load(path, identity, rows):
    return validate(read_checkpoint(path), identity, rows)


def commit(path, identity, choices, allocation):
    state = dict(identity=identity, choices=choices.copy(), allocation=list(allocation))
    validate(state, identity, choices.shape[-1])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, state)
