import copy
import numpy as np
import torch
from src.world_model import m3w_multiaction_cost as base
from src.world_model import m3w_policy_region_cost as api
from src.world_model import m3w_scene_context_cost as scene
from tests.test_m3w_scene_context_cost import scene_sample, SETTINGS


def regions_for(data, arm):
    n = len(data['truth']); choices = np.full(n, base.ARMS[arm][0])
    labels = data['truth'].numpy().copy()
    labels[~np.isin(np.arange(n), np.concatenate(data['groups']))] = np.nan
    return api.prepare_regions(labels, np.ones(n), np.array(['a']*n),
        np.array(['record']*n), choices, arm=arm)


def test_region_metadata_keeps_unknown_labels_out_of_teacher_centroids():
    y = np.ones((3, 7, 5)); y[-1] = np.nan
    regions = api.prepare_regions(y, np.ones(3), np.array(['a']*3), np.array(['r']*3),
        np.array([0, -1, 0]), arm='physical_only')
    assert regions['pool'].tolist() == [0, -1, -1]
    assert regions['selected_known_rows'] == 1 and regions['unknown_selected_rows'] == 1


def test_aggregate_moment_loss_has_nonvanishing_rare_harm_log_gradient():
    regions = dict(pool=torch.tensor([0, 0]), local_action=torch.tensor([0, 0]),
        target=torch.tensor([[0., .1, 1., 1., .1]]))
    def loss(which):
        raw = torch.tensor(-20., requires_grad=True)
        p = torch.ones((2, 1, 5)); p[:, :, 4] = raw.exp()
        if which == 'region':
            value = api.region_loss(p, np.arange(2), regions, torch.ones(5))
        else:
            value = (p[:, :, 4]-.1).square().mean()
        value.backward(); return abs(float(raw.grad))
    assert loss('region') > 10000*loss('square')


def test_zero_region_target_has_finite_gradient():
    p = torch.ones((2, 1, 5), requires_grad=True)
    regions = dict(pool=torch.tensor([0, 0]), local_action=torch.tensor([0, 0]),
        target=torch.zeros((1, 5)))
    loss = api.region_loss(p, np.arange(2), regions, torch.ones(5))
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_neutral_and_exact_resume_all_banks_without_neural_leakage(tmp_path):
    torch.set_num_threads(1); *_, data = scene_sample(); states = []
    kw = dict(data=data, settings=SETTINGS, seed=17, identity={'fixture':True})
    regions = regions_for(data, 'physical_neural')
    with scene.installed():
        old = base.fit(**kw, arm='physical_neural', path=tmp_path/'old.gz')
    disabled = api.fit(**kw, arm='physical_neural', path=tmp_path/'disabled.gz', regions=regions, region_weight=0)
    api.assert_neutral(old, disabled)
    for arm in base.ARMS:
        r = regions_for(data, arm)
        kwargs = dict(**kw, arm=arm, regions=r, region_weight=1.)
        direct = api.fit(**kwargs, path=tmp_path/(arm+'.gz'))
        api.fit(**kwargs, path=tmp_path/(arm+'_resume.gz'), stop_at=2)
        recovered = api.fit(**kwargs, path=tmp_path/(arm+'_resume.gz'), resume=True)
        for key in direct:
            if key != 'seconds':
                base.exact(direct[key], recovered[key])
        states.append(direct)
    changed = copy.deepcopy(data); changed['z'].motion[:, 6] += 1e5
    changed['z'].scene[:, 6] += 1e5; changed['truth'][:, 6] += 1e5
    other = api.fit(changed, arm='physical_only', settings=SETTINGS, seed=17,
        identity={'fixture':True}, regions=regions_for(changed, 'physical_only'), region_weight=1., path=tmp_path/'inactive.gz')
    base.exact(states[0]['model'], other['model'])
