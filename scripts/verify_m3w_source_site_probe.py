"""Verify source-site budgets, paired training and immutable completed resume."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_site_probe import SiteCorpus, load_config
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);out=ROOT/reg['output'];reports=ROOT/reg['reports']
    rp=reports/'report.json';report=json.loads(rp.read_text())
    assert report['completed_models']==30 and report['optimizer_updates']==60000
    data=SiteCorpus(reg)
    lookup={(t['arm'],t['site'],t['seed']):t for t in report['trials']};assert len(lookup)==30
    checks=[];exposure=[];steps=0
    for site in reg['sites']:
        train,weights,held,_=data.source_design(site)
        for seed in reg['seeds']:
            states=[]
            for arm in reg['arms']:
                t=lookup[arm,site,seed]
                for kind in ('checkpoint','prediction'):
                    assert file_digest(ROOT/t[kind+'_path'])==t[kind+'_sha256']
                cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
                assert cp['identity']==t['identity'] and cp['config']==reg['training'] and cp['arm']==arm
                assert cp['identity']['registration_sha256']==file_digest(args.registration)
                assert cp['step']==2000 and cp['draw_counts'].sum()==128000
                np.testing.assert_array_equal(cp['train_ids'],train)
                assert min(cp['train_ids'])>=data.nmain and not np.intersect1d(train,held).size
                assert all(torch.isfinite(v).all() for v in cp['model'].values())
                assert all(np.isfinite(v['bce']) and np.isfinite(v['gradient_norm']) for v in cp['losses'])
                counts=cp['draw_counts']
                exposure.append(dict(trial=t['trial'],eligible_rows=len(counts),sampled_rows=int((counts>0).sum()),
                    total_draws=int(counts.sum()),mean_draws_per_row=float(counts.mean()),
                    min_draws=int(counts.min()),max_draws=int(counts.max()),independent_sample_count=False))
                with np.load(ROOT/t['prediction_path'],allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['held_indices'],held)
                    assert np.isfinite(p['probability']).all()
                    assert ((p['probability']>=0)&(p['probability']<=1)).all()
                states.append(cp);steps+=cp['step']
            for key in ('train_ids','draw_counts'):
                np.testing.assert_array_equal(states[0][key],states[1][key])
            for key in ('sampler_rng','torch_rng'):
                assert torch.equal(states[0][key],states[1][key])
            for key in ('training_rows_sha256','normalizer_sha256','held_rows_sha256'):
                assert states[0]['identity'][key]==states[1]['identity'][key]
            checks.append(dict(site=site,seed=seed,same_train_rows=True,same_draw_counts=True,
                same_final_sampler_state=True,same_normalizer=True,finite_training=True,
                source_only_training=True,held_site_excluded=True))
    assert steps==60000
    paths=sorted([*out.glob('checkpoints/*.pt'),*out.glob('predictions/*.npz'),*out.glob('trials/*.json'),out/'identity.json'])
    assert len(paths)==91
    before={str(p.relative_to(ROOT)):file_digest(p) for p in paths};rh=file_digest(rp)
    run=subprocess.run([sys.executable,'scripts/run_m3w_source_site_probe.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    last=[json.loads(x) for x in run.stdout.splitlines() if x.startswith('{')][-1]
    assert last['state']=='completed_resume_verified' and last['new_updates']==last['new_fits']==0
    assert before=={str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(rp)==rh
    result=dict(result_source='fresh_run_verification_no_training',registration_sha256=file_digest(args.registration),
        report_sha256=rh,verified_fits=30,verified_torch_updates=steps,paired_stream_checks=checks,
        training_exposure=exposure,immutable_artifacts=91,all_hashes_unchanged=True,new_fits=0,new_updates=0,
        artifacts=before,artifact_manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        completed_resume_event=last,prediction_replay_is_separate=True,independent_confirmation=False,
        excluded_mutable_outputs=['heartbeat.json','support_audit.json'])
    json_write(reports/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','paired_stream_checks','training_exposure')}))


if __name__=='__main__':
    main()
