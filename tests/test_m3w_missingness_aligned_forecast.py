import torch

from src.world_model.m3w_easy_aligned_forecast import objective as easy
from src.world_model.m3w_missingness_aligned_forecast import objective


def test_unknown_loss_uses_only_path_disagreement_and_excludes_partial_labels():
    cv=torch.zeros(3,12,2);p=cv.clone();p[:,6,0]=3.;p.requires_grad_();y=cv.clone()
    m=torch.zeros(3,12,dtype=torch.bool);m[0]=True;m[1,0]=True
    f=torch.ones(3);a,_,_=easy(p,y,m,f,cv,2.,aligned=True)
    b,t=objective(p,y,m,f,cv,2.)
    assert b-a==t['unknown_radius_penalty']==1.
    (b-a).backward()
    assert not p.grad[:2].any() and p.grad[2,6,0]>0
    y[~m]=1234.
    assert objective(p,y,m,f,cv,2.)[0]==b
