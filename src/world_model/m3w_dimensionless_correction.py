"""Untrained unit-consistent fraction candidate; all frozen models stay untouched."""
import platform
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before Torch import')
from torch import nn
from src.world_model.m3w_partial_context import PartialContextSourceForecaster,condition_partial_inputs
from src.world_model.m3w_agent_track_context import AgentTrackContextForecaster
from src.world_model.m3w_baseline_relative_forecaster import BaselineRelativeForecaster
from src.world_model.m3w_european_source_forecast import BASELINES


class DimensionlessPartialConditioner(nn.Module):
    """Core output is a correction fraction; the outer causal budget supplies units.

    The inherited one-unit input clamp is unchanged. Scale equivariance is only
    claimed while it is inactive, not for arbitrary near-zero coordinate units.
    """
    def __init__(self,predictor):
        super().__init__(); self.predictor=predictor

    def forward(self,inputs):
        conditioned,_=condition_partial_inputs(inputs)
        return self.predictor(conditioned)


class DimensionlessAgentTrackSourceForecaster(PartialContextSourceForecaster):
    """Same grouped core, parameters and floor; remove only pre-squash scale restoration."""
    def __init__(self,baseline_index,*,width=64,heads=4,layers=2):
        nn.Module.__init__(self)
        if not 0<=baseline_index<len(BASELINES): raise ValueError('Declared causal baseline required')
        self.baseline_index=baseline_index
        core=AgentTrackContextForecaster(width=width,heads=heads,layers=layers)
        self.predictor=BaselineRelativeForecaster(DimensionlessPartialConditioner(core),mode='motion_bounded')
