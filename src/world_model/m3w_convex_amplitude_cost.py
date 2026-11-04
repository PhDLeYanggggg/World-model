"""Matched hinge-mixture harm curves, not a calibrated safety guarantee."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from src.world_model import m3w_amplitude_cost as parent

ALPHA=[0.,1/32,1/16,1/8,1/4,1/2,1.]
OriginalHead=parent.base.base.CostHead


class ShapeHead(OriginalHead):
    def __init__(self,*args,shape_arm,**kwargs):
        super().__init__(*args,**kwargs)
        if shape_arm not in ('unconstrained','convex'):
            raise ValueError('Fixed raw-control or convex decoder required')
        self.shape_arm=shape_arm

    def forward(self,x,envelope,segments,count):
        if self.shape_arm=='unconstrained':
            return super().forward(x,envelope,segments,count)
        if x.ndim!=3 or x.shape[1]!=7 or envelope.shape!=x.shape[:2]or segments.shape!=(len(x),):
            raise ValueError('Complete causal query batches required')
        code=torch.eye(7,dtype=x.dtype).expand(len(x),-1,-1)
        hidden=F.silu(self.encoder(torch.cat((x,code),-1)))
        if self.arm=='context':
            membership=F.one_hot(segments,count).to(hidden.dtype).T
            total=membership @ hidden.flatten(1)
            sizes=membership.T @ membership.sum(1)
            others=((membership.T @ total).reshape_as(hidden)-hidden)/(sizes-1).clamp_min(1)[:,None,None]
            hidden=hidden+others
        hidden=F.silu(hidden)
        raw=self.action(hidden)
        ref=self.reference(hidden.mean(1))
        reference=F.softplus(ref[:,0])[:,None].expand(-1,7)
        easy_reference=reference*torch.sigmoid(ref[:,1,None])
        fractions=F.softmax(torch.cat((raw[...,:2],torch.zeros_like(raw[...,:1])),-1),-1)
        # The small amplitude grid needs stable linear hinge arithmetic.
        alpha=torch.tensor(ALPHA,dtype=torch.float64,device=x.device)
        knots=alpha[:-1]
        basis=F.relu(alpha[:,None]-knots[None,:])/(1-knots)[None,:]
        weights=F.softmax(raw[:,1:,1],-1).to(torch.float64)
        weights=weights/weights.sum(1,keepdim=True)
        cap=(envelope[:,6]*fractions[:,6,1]).to(torch.float64)
        harm=cap[:,None]*(weights @ basis.T)
        easy_harm=cap[:,None]*((weights*torch.sigmoid(raw[:,1:,2]).to(torch.float64)) @ basis.T)
        benefit=torch.minimum(torch.minimum(envelope*fractions[...,0],reference),(envelope-harm).clamp_min(0))
        result=torch.stack((benefit,harm,reference,easy_reference,easy_harm),-1)
        # Preserve the existing FP32 loss/reduction; inference retains hinge precision.
        return result.to(x.dtype)if self.training else result


@contextmanager
def factory(arm):
    if parent.base.base.CostHead is not OriginalHead:
        raise RuntimeError('Single-process matched factory scope required')
    parent.base.base.CostHead=lambda *a,**kw:ShapeHead(*a,shape_arm=arm,**kw)
    try:
        yield
    finally:
        parent.base.base.CostHead=OriginalHead


def fit(data,*,identity,**kwargs):
    with factory('convex'):
        return parent.fit(data,arm='amplitude_bank',identity=dict(identity,
            shape_arm='convex',shape_schema='nested_nonnegative_hinges_v1'),**kwargs)


def predict(state,causal):
    arm=state['identity'].get('shape_arm','unconstrained')
    if arm=='convex'and state['identity'].get('shape_schema')!='nested_nonnegative_hinges_v1':
        raise ValueError('Frozen convex amplitude schema required')
    with factory(arm):
        return parent.predict(state,causal)
