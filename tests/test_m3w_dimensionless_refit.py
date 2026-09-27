import numpy as np
import torch
from src.evaluation import m3w_dimensionless_refit as metrics
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_native_forecast import fit_trial, fold_design, pack_geometry
from test_m3w_agent_track_context import geometry


def test_cost_direction_and_gate_guard():
    m = metrics.slice_metrics(np.array([1.,3.]), np.array([2.,2.]), np.zeros(2), np.zeros(2), np.ones(2,bool))
    assert m['gain_vs_grouped_percent']==0 and m['gain_vs_CV_percent'] is None
    assert m['absolute_harm_vs_grouped']==0 and m['mean_positive_harm_vs_grouped']==.5
    def d(point,lo):
        return dict(point=point,ci95=[lo,point+1])
    doc = {'summaries':{k:{'gain_vs_grouped_percent':v} for k,v in [
        ('ADE_all',d(0,-1)),('ADE_positive_easy',d(1,0)),('ADE_hard',d(1,0))]}}
    assert not metrics.gates(doc,2)['exploratory_forecaster_benefit']
    doc['summaries']['ADE_all']['gain_vs_grouped_percent']=d(3,1)
    assert metrics.gates(doc,2)['exploratory_forecaster_benefit']
    doc['summaries']['ADE_positive_easy']['gain_vs_grouped_percent']=d(-3,-4)
    assert not metrics.gates(doc,2)['exploratory_forecaster_benefit']
    assert not metrics.gates(doc,2)['deployment_changed']


def test_dimensionless_resume_and_label_poisoning(tmp_path):
    torch.set_num_threads(2)
    g = np.tile(geometry()[:1],(12,1))
    data = dict(geometry=g,target=g[:,332:356].reshape(12,12,2).copy()+1.0,
        valid=np.ones((12,12),bool),scale=np.ones(12),sites=np.repeat(['a','b','c'],4))
    fold = fold_design(data,'c','native_coordinate')
    settings = dict(steps=4,batch_size=4,learning_rate=.001,minimum_lr_ratio=.01,
        weight_decay=.0001,gradient_clip=5,checkpoint_every=2,heartbeat_every=1)
    def fit(home,resume=False,stop=None):
        torch.manual_seed(17)
        model = DimensionlessAgentTrackSourceForecaster(1,width=8,heads=2)
        report = fit_trial(model,data,fold,seed=17,settings=settings,identity={'synthetic':True},
            directory=home,resume=resume,stop_at=stop,heartbeat=lambda **kw:None)
        return model,report
    full,_ = fit(tmp_path/'full')
    fit(tmp_path/'resume',stop=2)
    resumed,report = fit(tmp_path/'resume',resume=True)
    assert report['new_updates']==2 and report['held_rows_sampled']==0
    for k,v in full.state_dict().items():
        torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    with torch.no_grad():
        a = resumed(pack_geometry(g))
    data['target'][:]=float('nan')
    data['valid'][:]=False
    with torch.no_grad():
        b = resumed(pack_geometry(g))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
