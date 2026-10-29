import copy
import torch
from src.world_model import m3w_policy_region_cost as old
from src.world_model import m3w_balanced_region_cost as api
from tests.test_m3w_scene_context_cost import scene_sample, SETTINGS
from tests.test_m3w_policy_region_cost import regions_for


def test_disabled_loss_exact_and_complete_region_resume_all_banks(tmp_path):
    torch.set_num_threads(1);*_,data=scene_sample()
    kwargs=dict(data=data,settings=SETTINGS,seed=17,identity={'fixture':True})
    regions=regions_for(data,'physical_neural')
    reference=old.fit(**kwargs,arm='physical_neural',path=tmp_path/'reference.gz',regions=regions,region_weight=0)
    disabled=api.fit(**kwargs,arm='physical_neural',path=tmp_path/'disabled.gz',regions=regions,region_weight=0,chunk_size=3)
    old.assert_neutral(reference,disabled)
    for arm in old.base.ARMS:
        params=dict(**kwargs,arm=arm,regions=regions_for(data,arm),region_weight=1.,chunk_size=3)
        direct=api.fit(**params,path=tmp_path/(arm+'.gz'))
        api.fit(**params,path=tmp_path/(arm+'_resume.gz'),stop_at=2)
        recovered=api.fit(**params,path=tmp_path/(arm+'_resume.gz'),resume=True)
        for key in direct:
            if key!='seconds':old.base.exact(direct[key],recovered[key])
        assert direct['auxiliary_teacher_rows']>0
        assert direct['auxiliary_forward_rows']==2*direct['auxiliary_teacher_rows']
        assert direct['draw_hash']==disabled['draw_hash']


def test_inactive_neural_action_cannot_change_physical_training(tmp_path):
    torch.set_num_threads(1);*_,data=scene_sample()
    kwargs=dict(settings=SETTINGS,seed=17,identity={'fixture':True},arm='physical_only',region_weight=1.,chunk_size=4)
    before=api.fit(data,**kwargs,regions=regions_for(data,'physical_only'),path=tmp_path/'before.gz')
    changed=copy.deepcopy(data);changed['z'].motion[:,6]+=1e5
    changed['z'].scene[:,6]+=1e5;changed['truth'][:,6]+=1e5
    after=api.fit(changed,**kwargs,regions=regions_for(changed,'physical_only'),path=tmp_path/'after.gz')
    old.base.exact(before['model'],after['model'])
