"""Verify fixed centering fits, exact replay, matched draws and completed resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_temporal_centered import load_config,setup
from scripts.verify_m3w_source_motion_quality import preserve_verification
from scripts.run_m3w_source_crossfit import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_source_crossfit import assemble_oof,cost_labels
from src.world_model.m3w_source_temporal_centered import ARMS,center_observed_tokens
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);parent,data,cached,reference=setup(reg)
    private,public=ROOT/reg['output'],ROOT/reg['reports']
    report=json.loads((public/'report.json').read_text());replay=json.loads((public/'replay.json').read_text())
    assert report['models']==len(set(replay['exact_replays']))==24 and report['optimizer_updates']==240000
    before={};keys=set();pieces={(arm,seed):[] for arm in ARMS for seed in reg['seeds']}
    for trial in report['trials']:
        key=(trial['site'],trial['seed'],trial['arm']);assert key not in keys;keys.add(key)
        train,_,held,outer,scale=data.configure(trial['site'])
        fold=trial['identity']['fold']
        assert array_hash(train)==fold['training_ids_sha256'] and array_hash(held)==fold['held_ids_sha256']
        assert array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant'])==fold['normalizer_sha256']
        original=next(t for t in reference['trials'] if t['site']==key[0] and t['seed']==key[1] and t['arm']=='sequence')
        for kind in ('checkpoint','prediction'):
            path=ROOT/trial[kind+'_path'];assert file_digest(path)==trial[kind+'_sha256']
            before[str(path.relative_to(ROOT))]=file_digest(path)
        saved=torch.load(ROOT/trial['checkpoint_path'],map_location='cpu',weights_only=False)
        parent_cp=torch.load(ROOT/original['checkpoint_path'],map_location='cpu',weights_only=False)
        assert saved['identity']==trial['identity'] and saved['step']==10000 and saved['scale']==scale
        assert saved['config']==reg['training'] and not saved['suppress_zero_targets']
        assert sum(v.numel() for v in saved['model'].values())==trial['parameters']==63960
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        np.testing.assert_array_equal(saved['train_ids'],train)
        np.testing.assert_array_equal(saved['draw_counts'],parent_cp['draw_counts'])
        assert saved['draw_counts'].sum()==640000 and torch.equal(saved['sampler_rng'],parent_cp['sampler_rng'])
        assert not np.intersect1d(train,held).size and not np.intersect1d(train,outer).size
        with np.load(ROOT/trial['prediction_path'],allow_pickle=False) as a:
            np.testing.assert_array_equal(a['train_ids'],train);np.testing.assert_array_equal(a['held_ids'],held)
            pred=a['held_prediction'].copy()
        radius=data.radius[held-data.nmain];support=data.support[held-data.nmain]
        assert np.isfinite(pred).all() and not pred[~support].any()
        assert np.all(np.linalg.norm(pred.astype(float),axis=-1)<=radius[:,None]*1.00001+1e-7)
        pieces[(key[2],key[1])].append(dict(ids=held,prediction=pred,cost_scale=np.full(len(held),scale)))
    assert len(keys)==24
    for item in report['oof_labels']:
        prediction,scale=assemble_oof(cached.ids,pieces[(item['arm'],item['seed'])])
        labels=cost_labels(prediction,data.target[cached.ids-data.nmain],scale)
        path=ROOT/item['path'];assert file_digest(path)==item['sha256']
        with np.load(path,allow_pickle=False) as a:
            for k,v in dict(ids=cached.ids,prediction=prediction,cost_scale=scale,**labels).items():
                np.testing.assert_array_equal(a[k],v)
        before[item['path']]=item['sha256']
    transform_rows=0
    for begin in range(0,len(cached.ids),128):
        rows=cached.rows[begin:begin+128]
        e=torch.from_numpy(cached.embedding[rows]);c=torch.from_numpy(cached.coverage[rows])
        for mode in ARMS:
            actual=center_observed_tokens(e,c,mode)
            torch.testing.assert_close(actual,center_observed_tokens(e+.25,c,mode),atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(actual[:1],center_observed_tokens(e[:1],c[:1],mode),atol=0,rtol=0)
            assert torch.isfinite(actual).all()
        transform_rows+=len(e)
    for path in [*private.joinpath('trials').glob('*.json'),private/'identity.json',
                 *[public/(n+'.json') for n in ('input_checks','report','replay','analysis')]]:
        before[str(path.relative_to(ROOT))]=file_digest(path)
    run=subprocess.run([sys.executable,'scripts/run_m3w_source_temporal_centered.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    event=[json.loads(s) for s in run.stdout.splitlines() if s.startswith('{')][-1]
    assert event['state']=='training_complete' and event['new_updates']==0
    assert before=={p:file_digest(ROOT/p) for p in before}
    result=dict(result_source='fresh_run_real_artifact_verification',registration_sha256=file_digest(args.registration),
        exact_replayed_heads=24,matched_sampling_streams=24,new_training_updates=240000,
        oof_archives_recomputed=6,real_transform_invariance_rows=transform_rows,
        immutable_artifacts=len(before),artifact_hashes=before,completed_resume=event,new_updates_on_resume=0,
        main_outer_rows_scored=0,new_deployment=False,sensor_asof_certified=False)
    preserve_verification(public/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))


if __name__=='__main__':main()
