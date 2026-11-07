import numpy as np

from src.evaluation.m3w_recording_bank_ceiling import solve


def test_bank_can_fix_easy_support_without_picking_two_paths_for_one_row():
    floor=np.zeros((2,12,2)); a=floor.copy(); b=floor.copy(); y=floor.copy()
    y[:,:,0]=np.array([1.,5.])[:,None]; a[:,:,0]=np.array([-1.,4.])[:,None]
    b[:,:,0]=np.array([.5,3.])[:,None]; m=np.ones((2,12),bool); w=np.ones(2)
    alone=solve(floor,np.stack((a,)),y,m,w,'r',2.,4.)
    both=solve(floor,np.stack((a,b)),y,m,w,'r',2.,4.)
    assert not alone['feasible']
    assert both['feasible'] and both['completion_gain_mass']==4.5
    assert both['action_counts']==[1,1]
