"""Matched temporal readouts of quality-only versus observed flow tokens."""
import numpy as np
import torch

from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics, embedding_lookup
from src.world_model.m3w_source_box_motion import motion_tokens, fit_motion_normalizer, token_payload


def supported_motion_tokens(raw, rotation, native_scale, radius, support):
    if (support.dtype != bool or support.shape != radius.shape or np.any(radius < 0)
            or np.any(support & (radius <= 0))):
        raise ValueError('Supported restoration rows need positive past radius')
    result = motion_tokens(raw, rotation, native_scale, np.where(support, radius, 1.))
    result[~support, :, :10] = 0
    return result


class BoxMotionDynamics(TemporalSourceDynamics):
    def __init__(self, gain):
        if not np.isfinite(gain) or gain < 1:
            raise ValueError('Positive registered train-only readout gain required')
        super().__init__('sequence')
        self.register_buffer('readout_gain', torch.tensor(float(gain), dtype=torch.float64))
        self.head[-1].register_forward_hook(self._condition)

    def _condition(self, module, inputs, output):
        return output/self.readout_gain.to(dtype=output.dtype, device=output.device)


class BoxMotionCorpus:
    def __init__(self, data, cached, archive):
        self.data, self.cached = data, cached
        with np.load(archive, allow_pickle=False) as a:
            self.ids = a['ids'].copy(); raw = a['features'].copy()
        np.testing.assert_array_equal(self.ids, cached.ids)
        self.lookup = np.repeat(np.arange(len(self.ids))[:, None], 8, 1)
        loc = self.ids-data.nmain
        self.tokens = supported_motion_tokens(raw, data.rotation[loc], data.native_scale[loc],
                                             data.radius[loc], data.support[loc])
        self.normalizer = None

    def configure(self, train):
        # Guard all rows before fitting even unsupervised feature statistics.
        self.cached.inputs(train, training=True)
        rows = embedding_lookup(train, self.ids, self.lookup)[:, 0]
        self.normalizer = fit_motion_normalizer(self.tokens[rows])
        return self.normalizer

    def inputs(self, ids, arm, *, training=False):
        if self.normalizer is None:
            raise ValueError('Train-only normalization must be fitted first')
        old, frame = self.cached.inputs(ids, training=training)
        rows = embedding_lookup(ids, self.ids, self.lookup)[:, 0]
        payload = token_payload(self.tokens[rows], self.normalizer, arm)
        return (old[0], torch.from_numpy(payload), old[2]), frame
