"""Passive capture of a frozen forecaster's past-context interaction state."""
import torch

from src.world_model.m3w_eqmotion_adapter import INPUT_KEYS


class InteractionCapture:
    def __init__(self, model, *, suffix='core.gcl_2', dimension=64):
        found=[(name, layer) for name,layer in model.named_modules() if name==suffix or name.endswith('.'+suffix)]
        if len(found)!=1 or dimension!=64:
            raise ValueError('Exactly one fixed pre-forecast64-dimensional interaction layer required')
        self.model=model;self.name,self.layer=found[0];self.dimension=dimension
        self.handle=None;self.values=[]

    def __enter__(self):
        if self.handle is not None: raise ValueError('Capture already active')
        def hook(module, inputs, output):
            h=output[0]
            if h.ndim!=3 or h.shape[-1]!=self.dimension or not torch.isfinite(h).all():
                raise ValueError('Finite per-agent interaction state required')
            self.values.append(h[:,0].detach().clone())
        self.handle=self.layer.register_forward_hook(hook)
        return self

    def run(self, inputs):
        if self.handle is None or set(inputs)!=INPUT_KEYS:
            raise ValueError('Active passive capture and declared past-only input fields required')
        self.values=[]
        with torch.no_grad(): prediction=self.model(inputs)
        if len(self.values)!=1: raise ValueError('One fixed interaction execution per forward required')
        hidden=self.values.pop()
        if hidden.shape!=(len(prediction),64): raise ValueError('Ego-row latent alignment mismatch')
        return prediction,hidden

    def __exit__(self, *exc):
        self.handle.remove();self.handle=None;self.values=[]
