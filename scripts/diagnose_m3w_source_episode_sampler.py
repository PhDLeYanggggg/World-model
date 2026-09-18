"""Post-hoc sampler-distribution diagnosis; no new selection or training."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_episode_sampler import load_config,setup,immutable_json
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    config=Path('configs/m3w_source_episode_sampler_v1.json');reg=load_config(config)
    _,data,cached,_=setup(reg);public=ROOT/reg['reports']
    report=json.loads((public/'report.json').read_text())
    audit=json.loads((ROOT/reg['event_audit']).read_text());ep=ROOT/audit['row_archive']['path']
    assert file_digest(ep)==audit['row_archive']['sha256']
    with np.load(ep,allow_pickle=False) as a: ids,groups=a['ids'].copy(),a['episodes'].copy()
    rows=[];folds=[]
    for site in reg['sites']:
        train,_,_,_,scale=data.configure(site)
        weights,support=episode_weights(train,ids,groups)
        target=data.target[train-data.nmain]
        cv=np.linalg.norm(target.astype(float),axis=-1).mean(1)
        changed=cv>0
        folds.append(dict(site=site,**support,
            uniform_nonzero_label_probability=float(changed.mean()),
            episode_nonzero_label_probability=float(weights@changed),
            weighted_cv_cost_over_original_scale=float(weights@cv/scale)))
        for t in report['trials']:
            if t['site']!=site:continue
            pp=ROOT/t['prediction_path'];cp=ROOT/t['checkpoint_path']
            assert file_digest(pp)==t['prediction_sha256'] and file_digest(cp)==t['checkpoint_sha256']
            with np.load(pp,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['train_ids'],train)
                error=np.linalg.norm(a['train_prediction'].astype(float)-target,axis=-1).mean(1)
            unweighted=100*(1-error.mean()/cv.mean())
            np.testing.assert_allclose(unweighted,t['training']['gain_percent'],atol=1e-9,rtol=0)
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            draws=saved['draw_counts'];assert draws.sum()==640000
            rows.append(dict(trial=t['trial'],site=site,arm=t['arm'],seed=t['seed'],
                actual_nonzero_draw_fraction=float(draws@changed/draws.sum()),
                unweighted_training_gain_percent=float(unweighted),
                episode_weighted_training_gain_percent=float(100*(1-weights@error/(weights@cv))),
                observed_draw_training_gain_percent=float(100*(1-draws@error/(draws@cv))),
                logged_gradient_clipped_fraction=float(np.mean([x['gradient_norm']>5 for x in t['fit']['trace']]))))
    summary={arm:dict(
        episode_weighted_training_gain_range=[min(x['episode_weighted_training_gain_percent'] for x in rows if x['arm']==arm),
                                             max(x['episode_weighted_training_gain_percent'] for x in rows if x['arm']==arm)],
        unweighted_training_gain_range=[min(x['unweighted_training_gain_percent'] for x in rows if x['arm']==arm),
                                       max(x['unweighted_training_gain_percent'] for x in rows if x['arm']==arm)])
        for arm in reg['arms']}
    result=dict(result_source='fresh_run_posthoc_diagnosis_of_frozen_fits',
        registration_sha256=file_digest(config), report_sha256=file_digest(public/'report.json'),
        folds=folds,trials=rows,summary=summary,
        future_labels_used_only_for_diagnosis=True,sampler_uses_future_labels=False,
        new_training=0,new_selection=False,new_deployment=False)
    immutable_json(public/'diagnosis.json',result)
    print(json.dumps(dict(folds=folds,summary=summary),indent=2))


if __name__=='__main__':main()
