"""Check all registered classifier artifacts and a zero-fit completed resume."""
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

import joblib
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg=json.loads(args.registration.read_text()); out=ROOT/reg['output']; reports=ROOT/reg['reports']
    rp=reports/'report.json'; report=json.loads(rp.read_text())
    assert report['complete'] and len(report['trials'])==45
    assert len({t['trial'] for t in report['trials']})==45
    assert sum(len(t['evaluation']) for t in report['trials'])==54
    checks=[]; steps=0
    for t in report['trials']:
        for field in ('model','prediction'):
            assert file_digest(ROOT/t[field+'_path'])==t[field+'_sha256']
        model=joblib.load(ROOT/t['model_path'])
        assert model['identity']==t['identity']
        assert t['identity']['registration_sha256']==file_digest(args.registration)
        assert np.isfinite(model['normalizer']['mean']).all()
        assert np.isfinite(model['normalizer']['std']).all()
        with np.load(ROOT/t['prediction_path'],allow_pickle=False) as prediction:
            p=prediction['probability']; ids=prediction['held_indices']
            assert len(p)==len(ids) and len(np.unique(ids))==len(ids)
            assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        if t['family']=='mlp':
            assert file_digest(ROOT/t['checkpoint_path'])==t['checkpoint_sha256']
            cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
            assert cp['identity']==t['identity'] and cp['config']==reg['models']['mlp']
            assert cp['step']==1000 and all(torch.isfinite(v).all() for v in cp['model'].values())
            assert all(np.isfinite(v['loss']) and np.isfinite(v['gradient_norm']) for v in cp['losses'])
            for key,value in model['model'].state_dict().items():
                assert torch.equal(value,cp['model'][key])
            steps+=cp['step']
        elif t['family']=='extra_trees':
            assert file_digest(ROOT/t['checkpoint_path'])==t['checkpoint_sha256']
            cp=joblib.load(ROOT/t['checkpoint_path'])
            assert cp['identity']==t['identity'] and cp['config']==reg['models']['extra_trees']
            assert cp['model'].n_estimators==len(cp['model'].estimators_)==256
            assert model['model'].n_estimators==256
        else:
            assert np.isfinite(model['model'].coef_).all()
        checks.append(dict(trial=t['trial'],artifacts_verified=True,finite_predictions=True))
    assert steps==15000
    paths=sorted([*out.glob('models/*.joblib'),*out.glob('predictions/*.npz'),*out.glob('trials/*.json'),
                  *out.glob('checkpoints/*.pt'),*out.glob('tree_checkpoints/*.joblib'),out/'identity.json'])
    assert len(paths)==166
    before={str(p.relative_to(ROOT)):file_digest(p) for p in paths}; rh=file_digest(rp)
    result=subprocess.run([sys.executable,'scripts/run_m3w_source_start_probe.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    last=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')][-1]
    assert last['state']=='completed_resume_verified' and last['new_fits']==last['new_updates']==0
    assert before=={str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(rp)==rh
    evidence=dict(result_source='fresh_run_verification_no_training',registration_sha256=file_digest(args.registration),
        report_sha256=rh,verified_fits=45,verified_torch_training_steps=steps,new_fits=0,new_updates=0,
        checks=checks,immutable_artifacts=len(paths),all_hashes_unchanged=True,artifacts=before,
        artifact_manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        completed_resume_event=last,independent_confirmation=False,prediction_replay_is_separate=True,
        excluded_mutable_outputs=['heartbeat.json','input_checks.json'])
    json_write(reports/'verification.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('checks','artifacts')}))


if __name__=='__main__':
    main()
