"""Verify paired sample streams, completed budgets and immutable visual resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True); args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg=json.loads(args.registration.read_text()); out=ROOT/reg['output']; reports=ROOT/reg['reports']
    rp=reports/'report.json'; report=json.loads(rp.read_text())
    assert report['complete'] and len(report['trials'])==30 and report['total_torch_updates']==60000
    assert sum(len(t['evaluation']) for t in report['trials'])==36
    lookup={(t['arm'],t['schedule'],t['seed'],t['fold']):t for t in report['trials']}
    assert len(lookup)==30
    checks=[]; steps=0; exposure=[]
    nmain=json.loads((reports/'input_checks.json').read_text())['main_rows']
    for schedule in reg['schedules']:
        for seed in reg['seeds']:
            for fold in ([-1] if schedule=='source_only' else [0,1]):
                states=[]
                for arm in reg['arms']:
                    t=lookup[arm,schedule,seed,fold]
                    for kind in ('checkpoint','prediction'):
                        assert file_digest(ROOT/t[kind+'_path'])==t[kind+'_sha256']
                    cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
                    assert cp['identity']==t['identity'] and cp['config']==reg['training'] and cp['arm']==arm
                    assert cp['identity']['registration_sha256']==file_digest(args.registration)
                    assert cp['step']==2000 and cp['draw_counts'].sum()==128000
                    assert all(torch.isfinite(v).all() for v in cp['model'].values())
                    assert all(np.isfinite(v['bce']) and np.isfinite(v['gradient_norm']) for v in cp['losses'])
                    for domain, keep in [('main',cp['train_ids']<nmain),('source',cp['train_ids']>=nmain)]:
                        counts=cp['draw_counts'][keep]
                        if len(counts):
                            exposure.append(dict(trial=t['trial'],domain=domain,eligible_rows=len(counts),
                                sampled_rows=int((counts>0).sum()),total_draws=int(counts.sum()),
                                mean_draws_per_row=float(counts.mean()),min_draws=int(counts.min()),
                                max_draws=int(counts.max()),independent_sample_count=False))
                    states.append(cp); steps+=cp['step']
                np.testing.assert_array_equal(states[0]['train_ids'],states[1]['train_ids'])
                np.testing.assert_array_equal(states[0]['draw_counts'],states[1]['draw_counts'])
                assert torch.equal(states[0]['sampler_rng'],states[1]['sampler_rng'])
                assert torch.equal(states[0]['torch_rng'],states[1]['torch_rng'])
                assert states[0]['identity']['training_rows_sha256']==states[1]['identity']['training_rows_sha256']
                assert states[0]['identity']['normalizer_sha256']==states[1]['identity']['normalizer_sha256']
                checks.append(dict(schedule=schedule,seed=seed,fold=fold,same_train_rows=True,
                    same_draw_counts=True,same_final_sampler_state=True,same_normalizer=True,finite_training=True))
    assert steps==60000
    paths=sorted([*out.glob('checkpoints/*.pt'),*out.glob('predictions/*.npz'),*out.glob('trials/*.json'),out/'identity.json'])
    assert len(paths)==91
    before={str(p.relative_to(ROOT)):file_digest(p) for p in paths}; rh=file_digest(rp)
    run=subprocess.run([sys.executable,'scripts/run_m3w_source_visual_start.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    last=[json.loads(x) for x in run.stdout.splitlines() if x.startswith('{')][-1]
    assert last['state']=='completed_resume_verified' and last['new_updates']==last['new_fits']==0
    assert before=={str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(rp)==rh
    result=dict(result_source='fresh_run_verification_no_training',registration_sha256=file_digest(args.registration),
        report_sha256=rh,verified_fits=30,verified_torch_updates=steps,paired_stream_checks=checks,
        training_exposure=exposure,
        immutable_artifacts=91,all_hashes_unchanged=True,new_fits=0,new_updates=0,artifacts=before,
        artifact_manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        completed_resume_event=last,prediction_replay_is_separate=True,independent_confirmation=False,
        excluded_mutable_outputs=['heartbeat.json','input_checks.json'])
    json_write(reports/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','paired_stream_checks','training_exposure')}))


if __name__=='__main__':
    main()
