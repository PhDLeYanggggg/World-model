from __future__ import annotations

import numpy as np

from src.models.jepa.stage18_context_encoder import Stage18ContextEncoder
from src.models.jepa.stage18_predictor import Stage18JEPAPredictor
from src.models.jepa.stage18_target_encoder import Stage18TargetEncoder


class SAMJEPA25D:
    """Self-Audited Multimodal JEPA 2.5D representation model.

    This is not autoregressive, not pixel reconstruction, not diffusion, and
    not latent generative rollout.
    """

    def __init__(self, weights, mean=None, std=None):
        self.context_encoder = Stage18ContextEncoder(mean=mean, std=std)
        self.target_encoder = Stage18TargetEncoder()
        self.predictor = Stage18JEPAPredictor(np.asarray(weights, dtype=np.float64))

    def predict_target_latent(self, context):
        return self.predictor.predict(self.context_encoder.encode(context))

