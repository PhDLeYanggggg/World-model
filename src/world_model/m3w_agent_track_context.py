"""Untrained grouped-history candidate; no change to frozen flat-token models."""
import platform
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import torch
from torch import nn
from src.world_model.m3w_supervised_intervention import PastContextForecaster
from src.world_model.m3w_partial_context import PartialContextSourceForecaster,PartialObservedConditioner
from src.world_model.m3w_baseline_relative_forecaster import BaselineRelativeForecaster
from src.world_model.m3w_context_conditioning import INPUT_KEYS
from src.world_model.m3w_european_source_forecast import BASELINES


class AgentTrackContextForecaster(PastContextForecaster):
    """Same two-layer weights: temporal within track, then interaction across tracks."""
    def __init__(self,*,width=64,heads=4,layers=2):
        if layers!=2:
            raise ValueError('Exactly one temporal and one interaction layer required')
        super().__init__(width=width,heads=heads,layers=layers,neighbor_policy='observed_tokens')

    def forward(self,inputs):
        if set(inputs)!=INPUT_KEYS:
            raise ValueError('Only declared past inputs and requested times accepted')
        h,n=inputs['history'],inputs['neighbors']
        hm,nm=inputs['history_mask'],inputs['neighbor_mask']
        values=torch.cat((h[:,None],n),1)
        valid=torch.cat((hm[:,None],nm),1)
        if (not hm.any(1).all() or not torch.isfinite(values[valid]).all()
                or (values[...,2][valid]>0).any()):
            raise ValueError('Finite observed past tokens and present ego required')
        batch,agents,steps,_=values.shape
        values=torch.where(valid[...,None],values,0.)
        tokens=self.embed(values)
        tokens[:,0]+=self.modality.weight[0]
        tokens[:,1:]+=self.modality.weight[1]
        # Empty padded agents need a numerical dummy key, never a visible agent.
        temporal_mask=valid.flatten(0,1).clone()
        empty=~temporal_mask.any(1)
        temporal_mask[empty,0]=True
        temporal=self.encoder.layers[0](tokens.flatten(0,1),src_key_padding_mask=~temporal_mask)
        temporal=temporal.reshape(batch,agents,steps,-1)
        pooled=torch.where(valid[...,None],temporal,0.).sum(2)/valid.sum(2).clamp_min(1)[...,None]
        agent_valid=valid.any(2)
        memory=self.encoder.layers[1](pooled,src_key_padding_mask=~agent_valid)
        if self.encoder.norm is not None: memory=self.encoder.norm(memory)
        request=inputs['request_mask']
        query_values=torch.cat((inputs['baseline'],inputs['prediction_time'][...,None]),-1)
        if not torch.isfinite(query_values[request]).all() or (inputs['prediction_time'][request]<=0).any():
            raise ValueError('Finite causal rollout and requested future times required')
        queries=self.query(torch.where(request[...,None],query_values,0.))
        decoded,_=self.attention(queries,memory,memory,key_padding_mask=~agent_valid,need_weights=False)
        return torch.where(request[...,None],self.output(decoded+queries),0.)


class AgentTrackSourceForecaster(PartialContextSourceForecaster):
    """Versioned topology candidate with identical parameter shapes and initial floor."""
    def __init__(self,baseline_index,*,width=64,heads=4,layers=2):
        nn.Module.__init__(self)
        if not 0<=baseline_index<len(BASELINES): raise ValueError('Declared causal baseline required')
        self.baseline_index=baseline_index
        core=AgentTrackContextForecaster(width=width,heads=heads,layers=layers)
        self.predictor=BaselineRelativeForecaster(PartialObservedConditioner(core),mode='motion_bounded')
