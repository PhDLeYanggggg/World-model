import numpy as np
import pytest
import torch

from scripts.run_m3w_moving_source_forecast import same_scientific_checkpoint


def test_checkpoint_comparison_ignores_only_elapsed_time_not_model_optimizer_or_sampler():
    a=dict(seconds=1.,step=100,model=dict(w=torch.tensor([1.,2.])),
           optimizer=dict(step=torch.tensor(100.)),draws=np.array([1,2]),
           sampler_rng=torch.tensor([1,2],dtype=torch.uint8),losses=[dict(step=100,loss=1.2)])
    same_scientific_checkpoint(a,dict(a,seconds=9.))
    with pytest.raises(ValueError):same_scientific_checkpoint(a,dict(a,step=101))
    with pytest.raises(ValueError):same_scientific_checkpoint(a,dict(a,model=dict(w=torch.tensor([2.,2.]))))
    with pytest.raises(AssertionError):same_scientific_checkpoint(a,dict(a,draws=np.array([1,3])))
