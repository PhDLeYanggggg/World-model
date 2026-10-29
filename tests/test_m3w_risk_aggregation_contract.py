"""TRAIN query-mean proxy and readout agent-mass risk are not equivalent."""
import numpy as np
import torch

from src.world_model.m3w_task_risk_objective import query_mean
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds


def test_query_mean_feasibility_does_not_prove_agent_mass_risk():
    # One low-risk singleton versus ten higher-risk agents in another query.
    y=np.array([[0.,1.,100.,100.,1.]]+[[0.,.22,1.,1.,.22]]*10)
    excess=torch.from_numpy(y[:,4]-.02*y[:,3])
    segments=torch.tensor([0]+[1]*10,dtype=torch.int64)
    assert query_mean(excess,segments,2)[0]<0
    assert excess.sum()>0
    bound=completion_bounds(y,np.ones(11,bool),np.full(11,2.))
    assert bound['easy_selected_risk_upper']>.02
    assert not bound['finite_completion_supported']
