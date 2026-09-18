import numpy as np
import pytest
import torch

from src.world_model.m3w_source_visual_start import VisualStart, fit_visual, probabilities


def payload():
    torch.manual_seed(3)
    geometry = torch.randn(7, 6)
    rgb = torch.rand(7, 8, 3, 32, 32)
    coverage = torch.rand(7, 8, 1, 32, 32)
    coverage[:, :2] = 0
    return geometry, rgb, coverage


def test_mask_control_and_unobserved_pixels_are_invariant():
    geometry, rgb, coverage = payload(); model = VisualStart(6)
    changed = torch.rand_like(rgb)*999
    assert torch.equal(model(geometry,rgb,coverage,'mask_only'), model(geometry,changed,coverage,'mask_only'))
    changed = torch.where(coverage == 0, changed, rgb)
    assert torch.equal(model(geometry,rgb,coverage,'past_rgb'), model(geometry,changed,coverage,'past_rgb'))
    with pytest.raises(TypeError):
        model(geometry,rgb,coverage,'past_rgb',future_endpoint=torch.ones(7,2))


def test_shape_and_arm_rejection():
    values = payload(); model = VisualStart(6)
    with pytest.raises(ValueError):
        model(*values,'current_rgb')
    with pytest.raises(ValueError):
        model(values[0],values[1][:,:7],values[2],'past_rgb')


def test_lazy_mixed_image_join_preserves_query_order_and_separates_labels():
    from scripts.run_m3w_source_visual_start import VisualCorpus
    data = object.__new__(VisualCorpus)
    def store(colors):
        return dict(rgb=np.broadcast_to(np.asarray(colors,dtype=np.uint8)[:,None,None,None],
                                       (len(colors),3,32,32)).copy(),
                    coverage=np.full((len(colors),32,32),9,np.uint8),
                    image_rows=np.repeat(np.arange(len(colors))[:,None],8,axis=1))
    data.main = store([0,1,2,3]); data.mid = np.array([1,3]); data.nmain = 2
    data.sid = np.array([2,0]); data.record_ids = np.array([1,0,0]); data.local_ids = np.array([0,0,1])
    data.images = [store([50,51]),store([100])]
    data.y = np.array([0,1,0,1]); data.z = np.zeros((4,476),np.float32)
    data.allowed = np.array([True,False,True,False])
    ids = np.array([3,0,2,1]); rgb,cov = data.raw_images(ids)
    np.testing.assert_array_equal(rgb[:,0,0,0,0],[100,1,51,3])
    assert np.all(cov == 9)
    before = data.inputs(ids); data.y = 1-data.y; after = data.inputs(ids)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    with pytest.raises(ValueError,match='Held'):
        data.inputs(np.array([1]),training=True)
    with pytest.raises(ValueError,match='Held'):
        data.loss_labels(np.array([1]))


def test_exact_resume_and_matched_image_arm_draws(tmp_path):
    torch.set_num_threads(2); values = payload(); y = (torch.arange(7)%2).float()
    ids = np.arange(7); weights = np.arange(1,8,dtype=float)/28
    cfg = dict(updates=7,batch_size=3,learning_rate=.001,weight_decay=.001,checkpoint_every=2)
    def run(name,arm='past_rgb',stop=None,identity=None):
        torch.manual_seed(17); model = VisualStart(6)
        result = fit_visual(model,lambda i:tuple(v[i] for v in values),lambda i:y[i],ids,weights,
            arm=arm,seed=17,config=cfg,identity=identity or {'fixed':1},checkpoint=tmp_path/name,
            heartbeat=lambda **_:None,stop_at=stop)
        return model,result
    a,_ = run('full.pt'); run('resume.pt',stop=3); b,result = run('resume.pt')
    assert result['new_updates'] == 4
    for key,value in a.state_dict().items():
        assert torch.equal(value,b.state_dict()[key])
    _,done = run('resume.pt'); assert done['new_updates'] == 0
    run('mask.pt','mask_only')
    full = torch.load(tmp_path/'full.pt',weights_only=False); mask = torch.load(tmp_path/'mask.pt',weights_only=False)
    np.testing.assert_array_equal(full['draw_counts'],mask['draw_counts'])
    assert torch.equal(full['sampler_rng'],mask['sampler_rng'])
    assert full['draw_counts'].sum() == 21
    with pytest.raises(ValueError,match='identity'):
        run('resume.pt',identity={'changed':1})
    np.testing.assert_array_equal(probabilities(a,lambda i:tuple(v[i] for v in values),ids,'past_rgb'),
                                  probabilities(b,lambda i:tuple(v[i] for v in values),ids,'past_rgb'))
