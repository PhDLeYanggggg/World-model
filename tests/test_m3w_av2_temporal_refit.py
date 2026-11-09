import numpy as np
import torch

from scripts import run_m3w_av2_temporal_refit as run


def test_crossed_decoding_uses_the_requested_checkpoint_and_bank():
    class Fixed:
        def __init__(self, gain):
            self.gain = gain

        def __call__(self, x):
            z = torch.zeros(len(x), 6)
            z[:, 0] = torch.tensor(self.gain)
            return z

    m = dict(x=np.zeros((4, 1), np.float32), opts=np.array([[0, 1], [2, 3]]),
             unit=np.ones(4), row=np.arange(4))
    keep = dict(base=np.array([True, False]), union=np.array([True, True]))
    nets = dict(base=Fixed([1., 1., 3., 3.]), union=Fixed([3., 3., 1., 1.]))
    picks, confidence = run.choices(nets, m, keep)
    assert picks == dict(refit_base=0, base_on_union=1, union_on_base=0, refit_union=0)
    assert all(0. < s < 1. for s in confidence.values())
