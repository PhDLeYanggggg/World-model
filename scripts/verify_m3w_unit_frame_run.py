"""Verify fixed unit-frame fits, sample matching and zero-update completed resume."""
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
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_unit_frame_training import ARMS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg=json.loads(args.registration.read_text())
    out,reports=ROOT/reg['output'],ROOT/reg['reports']
    report_path=reports/'report.json'; report=json.loads(report_path.read_text())
    if not report['complete'] or report['new_fits']!=27 or len(report['trials'])!=36:
        raise ValueError('Complete 27 new and nine cached fits required')
    lookup={(t['arm'],t['seed'],t['fold']):t for t in report['trials']}
    if len(lookup)!=36:
        raise ValueError('Repeated trial key')
    checks=[]; steps=0
    for arm in ARMS:
        for seed in reg['seeds']:
            for fold in range(3):
                t=lookup[arm,seed,fold]; old=lookup['legacy_sdd_aux',seed,fold]
                for item in (t,old):
                    for kind in ('checkpoint','prediction'):
                        if file_digest(ROOT/item[kind+'_path'])!=item[kind+'_sha256']:
                            raise ValueError('Artifact changed')
                cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
                parent=torch.load(ROOT/old['checkpoint_path'],map_location='cpu',weights_only=False)
                assert cp['step']==6000 and cp['identity']==t['identity']
                assert cp['config']==reg['training'] and cp['arm']==arm
                assert cp['identity']['registration_sha256']==file_digest(args.registration)
                assert all(torch.isfinite(x).all() for x in cp['model'].values())
                assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in cp['losses'])
                np.testing.assert_array_equal(cp['draw_counts'][0],parent['draw_counts']['main'])
                np.testing.assert_array_equal(cp['draw_counts'][1],parent['draw_counts']['auxiliary'])
                assert torch.equal(cp['sampler_rng'],parent['sampler_rng'])
                assert cp['draw_counts'][0].sum()==256000 and cp['draw_counts'][1].sum()==128000
                checks.append(dict(trial=t['trial'],same_source_and_main_draw_counts=True,
                    same_final_main_sampler_state=True,finite_weights_losses_gradients=True))
                steps+=cp['step']
    assert steps==162000
    paths=sorted([*out.glob('checkpoints/*.pt'),*out.glob('predictions/*.npz'),*out.glob('trials/*.json'),out/'identity.json'])
    if len(paths)!=82:
        raise ValueError('Expected 82 immutable artifacts')
    before={str(p.relative_to(ROOT)):file_digest(p) for p in paths}; rh=file_digest(report_path)
    result=subprocess.run([sys.executable,'scripts/run_m3w_unit_frame_training.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    final=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')][-1]
    assert final['state']=='completed_resume_verified' and final['new_optimizer_updates']==0
    assert before=={str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(report_path)==rh
    evidence=dict(result_source='fresh_run_verification_no_training',registration_sha256=file_digest(args.registration),
        report_sha256=rh,verified_fits=27,verified_original_training_steps=steps,new_optimizer_updates=0,
        checks=checks,immutable_artifacts=82,all_hashes_unchanged=True,artifacts=before,
        artifact_manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        final_resume_event=final,independent_confirmation=False,
        prediction_replay_is_separate=True,excluded_mutable_outputs=['heartbeat.json','input_checks.json'])
    json_write(reports/'verification.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('checks','artifacts')}))


if __name__=='__main__':
    main()
