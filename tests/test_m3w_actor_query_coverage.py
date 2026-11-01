import numpy as np
import pytest
from scripts.diagnose_m3w_actor_query_coverage import reconstruct,parent


def test_query_coverage_matches_original_sampler_and_final_rng():
    keys=[('a','r',i) for i in range(11)]+[('b','q',i) for i in range(3)]
    blocks=[np.array([i]) for i in range(len(keys))];sources=[np.arange(11),np.arange(11,14)]
    rng=parent.api.torch.Generator().manual_seed(17+7919);counts=np.zeros(len(keys),int)
    for _ in range(50):
        _,_,ids=parent.api.parent.draw_queries(blocks,sources,8,rng);np.add.at(counts,ids,1)
    result,final=reconstruct(keys,seed=17,steps=50,batch_size=8)
    parent.api.parent.exact(rng.get_state(),final)
    for out,g in zip(result,sources):
        assert out['draws']==counts[g].sum()
        assert out['never_drawn']==(counts[g]==0).sum()
        assert out['seen_fraction']==(counts[g]>0).mean()


def test_invalid_sampling_design_is_not_accepted():
    with pytest.raises(ValueError):reconstruct([],seed=17,steps=10,batch_size=4)
    with pytest.raises(ValueError):reconstruct([('a','r',0),('b','r',0)],seed=17,steps=10,batch_size=3)
