"""Past detector tokens in the actual grouped, dimensionless SOURCE producer."""
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_context_conditioning import INPUT_KEYS
from src.world_model.m3w_detector_forecaster import (
    DetectorGeometry,pack_geometry,installed,ORIGINAL_PACK,REPRESENTATIONS,assert_neutral,assert_recovery,native)


class ActiveDetectorForecaster(DimensionlessAgentTrackSourceForecaster):
    def __init__(self,baseline_index,representation,*,width=64,heads=4,layers=2):
        if representation not in REPRESENTATIONS:raise ValueError('Declared detector representation required')
        super().__init__(baseline_index,width=width,heads=heads,layers=layers)
        self.representation=representation;self.detector_weight=nn.Parameter(torch.zeros(width,8))

    def forward(self,inputs):
        if set(inputs)!=(INPUT_KEYS|{'detector_features'}):raise ValueError('Declared past features only')
        plain={k:v for k,v in inputs.items() if k!='detector_features'};detector=inputs['detector_features']
        if detector.shape!=(len(plain['history']),8,8) or not torch.isfinite(detector).all():
            raise ValueError('Finite aligned eight-step detector tokens required')
        if self.representation=='disabled':return super().forward(plain)
        core=self.predictor.predictor.predictor
        def inject(module,args,output):
            if output.ndim!=4 or output.shape[:3]!=(len(detector),9,8):
                raise ValueError('Original ego-plus-eight-neighbor temporal topology required')
            delta=F.linear(detector,self.detector_weight)
            return torch.cat((output[:,:1]+delta[:,None],output[:,1:]),1)
        # Keep the original masking, grouped attention, conditioner and output units.
        hook=core.embed.register_forward_hook(inject)
        try:return super().forward(plain)
        finally:hook.remove()
