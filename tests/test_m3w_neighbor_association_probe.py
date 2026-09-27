import numpy as np
import torch
from src.evaluation.m3w_neighbor_association_probe import scramble_associations,neighbor_path_length
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_partial_context import PartialContextSourceForecaster


def test_cross_time_track_assignment_is_lost_by_flat_neighbor_tokens():
    g=np.zeros((1,476),np.float32)
    g[:,:16]=np.stack([np.arange(-7,1),np.zeros(8)],axis=-1).reshape(1,-1)
    g[:,16:24]=np.arange(-7,1)/12
    xy=g[:,38:166].reshape(1,8,8,2)
    xy[0,0,:,0]=np.arange(8)+10; xy[0,1,:,0]=np.arange(8)-10
    g[:,166:230].reshape(1,8,8)[:,:2]=np.arange(-7,1)/12
    g[:,230:294].reshape(1,8,8)[:,:2]=1
    changed=scramble_associations(g)
    assert neighbor_path_length(changed)[0]>neighbor_path_length(g)[0]
    np.testing.assert_array_equal(changed[:,:38],g[:,:38])
    np.testing.assert_array_equal(changed[:,166:],g[:,166:])
    torch.manual_seed(17)
    model=PartialContextSourceForecaster(1,width=8,heads=2,layers=1).eval()
    torch.nn.init.normal_(model.predictor.predictor.predictor.output[-1].weight,std=.02)
    with torch.no_grad():
        a=model(pack_geometry(g)); b=model(pack_geometry(changed))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(g[:,38:166].reshape(1,8,8,2)[:,:,7],
                                  changed[:,38:166].reshape(1,8,8,2)[:,:,7])
