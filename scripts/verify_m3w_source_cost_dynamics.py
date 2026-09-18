"""Check full budgets, matched streams, lineage and immutable completed resume."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_cost_dynamics import DynamicsCorpus, load_config
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);out=ROOT/reg['output'];reports=ROOT/reg['reports'];rp=reports/'report.json'
    report=json.loads(rp.read_text());assert report['completed_models']==60 and report['optimizer_updates']==120000
    data=DynamicsCorpus(reg);lookup={(t['objective'],t['arm'],t['site'],t['seed']):t for t in report['trials']};assert len(lookup)==60
    checks=[];exposure=[];steps=0
    for site in reg['sites']:
        train,w,held,_,normalizer=data.dynamics_design(site)
        for seed in reg['seeds']:
            states=[];gates={}
            for arm in reg['arms']:
                for objective in reg['objectives']:
                    t=lookup[objective,arm,site,seed]
                    for kind in ('checkpoint','prediction'):
                        assert file_digest(ROOT/t[kind+'_path'])==t[kind+'_sha256']
                    cp=torch.load(ROOT/t['checkpoint_path'],map_location='cpu',weights_only=False)
                    assert cp['identity']==t['identity'] and cp['config']==reg['training']
                    assert cp['arm']==arm and cp['objective']==objective and cp['normalizer']==normalizer
                    assert cp['step']==2000 and cp['draw_counts'].sum()==128000
                    np.testing.assert_array_equal(cp['train_ids'],train)
                    assert min(train)>=data.nmain and not np.intersect1d(train,held).size
                    assert all(torch.isfinite(v).all() for v in cp['model'].values())
                    assert all(all(np.isfinite(v[k]) for k in ('objective_loss','normalized_batch_ade','gradient_norm')) for v in cp['losses'])
                    with np.load(ROOT/t['prediction_path'],allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['held_indices'],held)
                        assert np.isfinite(p['prediction']).all() and p['prediction'].shape==(len(held),12,2)
                        gates[arm,objective]=p['gate'].copy()
                    counts=cp['draw_counts'];exposure.append(dict(trial=t['trial'],eligible_rows=len(counts),
                        sampled_rows=int((counts>0).sum()),total_draws=int(counts.sum()),mean_draws_per_row=float(counts.mean()),
                        min_draws=int(counts.min()),max_draws=int(counts.max())))
                    states.append(cp);steps+=cp['step']
                np.testing.assert_array_equal(gates[arm,'ade'],gates[arm,'log_ade'])
            for cp in states[1:]:
                for k in ('train_ids','draw_counts'):np.testing.assert_array_equal(states[0][k],cp[k])
                for k in ('sampler_rng','torch_rng'):assert torch.equal(states[0][k],cp[k])
                for k in ('normalizer_sha256','training_rows_sha256','held_rows_sha256'):
                    assert states[0]['identity'][k]==cp['identity'][k]
            checks.append(dict(site=site,seed=seed,four_matched_streams=True,same_normalizer=True,
                same_target_loss_scale=True,same_arm_gate_identical_between_objectives=True,
                finite_training=True,held_site_excluded=True))
    assert steps==120000
    paths=sorted([*out.glob('checkpoints/*.pt'),*out.glob('predictions/*.npz'),*out.glob('trials/*.json'),out/'identity.json'])
    assert len(paths)==181
    before={str(p.relative_to(ROOT)):file_digest(p) for p in paths};rh=file_digest(rp)
    run=subprocess.run([sys.executable,'scripts/run_m3w_source_cost_dynamics.py','--registration',str(args.registration)],cwd=ROOT,capture_output=True,text=True,check=True)
    last=[json.loads(x) for x in run.stdout.splitlines() if x.startswith('{')][-1]
    assert last['state']=='completed_resume_verified' and last['new_updates']==last['new_fits']==0
    assert before=={str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(rp)==rh
    result=dict(result_source='fresh_run_verification_no_training',report_sha256=rh,
        registration_sha256=file_digest(args.registration),verified_fits=60,verified_updates=steps,
        matched_four_way_checks=checks,training_exposure=exposure,immutable_artifacts=181,
        all_hashes_unchanged=True,new_fits=0,new_updates=0,artifacts=before,
        artifact_manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        completed_resume_event=last,prediction_replay_is_separate=True,independent_confirmation=False)
    json_write(reports/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','training_exposure','matched_four_way_checks')}))


if __name__=='__main__':main()
