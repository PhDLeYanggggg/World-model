"""Validate a cross-runtime scalar reduction while retaining frozen TRAIN state."""
from contextlib import contextmanager
import math


def validate_scale_only(reference, recomputed, exact):
    if set(reference) != set(recomputed) or 'scale' not in reference:
        raise ValueError('Preprocessing schema changed')
    for key in reference:
        a, b = reference[key], recomputed[key]
        if getattr(a, 'dtype', None) != getattr(b, 'dtype', None) or getattr(a, 'shape', None) != getattr(b, 'shape', None):
            raise ValueError('Preprocessing dtype/shape changed: '+key)
    exact({k:v for k,v in reference.items() if k!='scale'},
          {k:v for k,v in recomputed.items() if k!='scale'})
    a, b = reference['scale'], recomputed['scale']
    if type(a) is not float or type(b) is not float or not (math.isfinite(a) and math.isfinite(b) and a>0 and b>0):
        raise ValueError('Positive finite Python float64 scale required')
    if abs(a-b) > 4*max(math.ulp(a), math.ulp(b)):
        raise ValueError('TRAIN scale differs beyond four float64 ULPs')


@contextmanager
def frozen_preprocess(core, frozen):
    original = core.preprocess

    def recompute_and_validate(*args, **kwargs):
        fresh = original(*args, **kwargs)
        validate_scale_only(frozen, fresh, core.exact)
        # This value, not the recomputed reduction, remains the numerical input.
        return frozen

    core.preprocess = recompute_and_validate
    try:
        yield
    finally:
        core.preprocess = original
