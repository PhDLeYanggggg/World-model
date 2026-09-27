import numpy as np
import torch
from src.world_model.m3w_agent_track_context import AgentTrackSourceForecaster
from src.world_model.m3w_partial_context import PartialContextSourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_european_source_forecast import baseline_torch
from src.evaluation.m3w_neighbor_association_probe import scramble_associations


def geometry():
    g=np.zeros((2,476),np.float32)
    g[:,:16]=np.stack([np.arange(-7,1),np.zeros(8)],-1).reshape(1,-1)
    g[:,16:24]=np.arange(-7,1)/12
    xy=g[:,38:166].reshape(2,8,8,2)
    xy[:,0,:,0]=np.arange(8)+10; xy[:,1,:,0]=np.arange(8)-10
    g[:,166:230].reshape(2,8,8)[:,:2]=np.arange(-7,1)/12
    g[:,230:294].reshape(2,8,8)[:,:2]=1
    return g


def test_same_initial_weights_and_floor():
    torch.manual_seed(17); old=PartialContextSourceForecaster(1,width=8,heads=2)
    torch.manual_seed(17); new=AgentTrackSourceForecaster(1,width=8,heads=2)
    assert old.state_dict().keys()==new.state_dict().keys()
    for k,v in old.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
    x=pack_geometry(geometry())
    torch.testing.assert_close(new(x),baseline_torch(x['history'],1),atol=0,rtol=0)
    ((new(x)-1)**2).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())


def test_can_distinguish_track_associations_but_not_agent_renaming():
    torch.manual_seed(29); model=AgentTrackSourceForecaster(1,width=8,heads=2).eval()
    torch.nn.init.normal_(model.predictor.predictor.predictor.output[-1].weight,std=.2)
    g=geometry(); x=pack_geometry(g)
    renamed={k:v.clone() for k,v in x.items()}
    order=[1,0,2,3,4,5,6,7]
    renamed['neighbors']=renamed['neighbors'][:,order]
    renamed['neighbor_mask']=renamed['neighbor_mask'][:,order]
    with torch.no_grad():
        a=model(x); b=model(pack_geometry(scramble_associations(g))); c=model(renamed)
    assert not torch.allclose(a,b,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(a,c,rtol=1e-5,atol=1e-5)


def test_missing_agents_and_masked_values_are_safe():
    torch.manual_seed(43); model=AgentTrackSourceForecaster(1,width=8,heads=2).eval()
    torch.nn.init.normal_(model.predictor.predictor.predictor.output[-1].weight,std=.1)
    x=pack_geometry(geometry()); x['neighbor_mask'][0]=False
    with torch.no_grad(): a=model(x)
    x['neighbors'][~x['neighbor_mask']]=float('nan')
    with torch.no_grad(): b=model(x)
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
