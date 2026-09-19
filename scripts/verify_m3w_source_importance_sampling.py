"""Verify corrected heads, exact exposure controls and immutable resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_importance_sampling import load_config, context, ARMS
from scripts.run_m3w_source_crossfit import array_hash
from scripts.verify_m3w_source_motion_quality import preserve_verification
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_importance_sampling import uniform_risk_factors, OBJECTIVE
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration',type=Path,required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    _,data,cached,uniform,control,ids,groups,_ = context(reg)
    public,private = ROOT/reg['reports'],ROOT/reg['output']
    report = json.loads((public/'report.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    preflight = json.loads((public/'objective_check.json').read_text())
    assert preflight['registration_sha256'] == file_digest(args.registration)
    assert len(report['trials']) == len(set(replay['exact_replays'])) == 24
    assert report['optimizer_updates'] == 240000
    pieces = {(arm,seed):[] for arm in ARMS for seed in reg['seeds']}
    hashes,pairs,keys = {},{},set()
    for trial in report['trials']:
        key = trial['site'],trial['seed'],trial['arm']; assert key not in keys; keys.add(key)
        train,_,held,outer,scale = data.configure(key[0])
        probs,support = episode_weights(train,ids,groups); factors = uniform_risk_factors(probs)
        np.testing.assert_allclose(probs/probs.sum()*factors,np.full(len(train),1/len(train)),rtol=1e-12,atol=1e-18)
        identity = trial['identity']; fold = identity['fold']
        assert identity['sampler_sha256'] == array_hash(train,probs)
        assert identity['importance_sha256'] == array_hash(train,factors)
        assert all(identity[k] == v for k,v in support.items())
        assert array_hash(train) == fold['training_ids_sha256'] and array_hash(held) == fold['held_ids_sha256']
        assert array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant']) == fold['normalizer_sha256']
        assert not np.intersect1d(train,held).size and not np.intersect1d(train,outer).size
        for kind in ('checkpoint','prediction'):
            path = ROOT/trial[kind+'_path']; assert file_digest(path) == trial[kind+'_sha256']
            hashes[str(path.relative_to(ROOT))] = file_digest(path)
        saved = torch.load(ROOT/trial['checkpoint_path'],map_location='cpu',weights_only=False)
        assert saved['identity'] == identity and saved['step'] == 10000 and saved['scale'] == scale
        assert saved['config'] == reg['training'] and saved['objective'] == OBJECTIVE
        assert sum(v.numel() for v in saved['model'].values()) == trial['parameters'] == 63960
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        np.testing.assert_array_equal(saved['train_ids'],train)
        np.testing.assert_array_equal(saved['probabilities'],probs)
        np.testing.assert_array_equal(saved['importance_factors'],factors)
        assert saved['draw_counts'].sum() == 640000
        assert array_hash(saved['draw_counts']) == trial['sampled_draws_sha256']
        old = next(t for t in control['trials'] if t['trial'] == trial['trial'])
        old_saved = torch.load(ROOT/old['checkpoint_path'],map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(saved['draw_counts'],old_saved['draw_counts'])
        assert torch.equal(saved['sampler_rng'],old_saved['sampler_rng'])
        if key[:2] in pairs: np.testing.assert_array_equal(saved['draw_counts'],pairs[key[:2]])
        pairs[key[:2]] = saved['draw_counts'].copy()
        with np.load(ROOT/trial['prediction_path'],allow_pickle=False) as a:
            np.testing.assert_array_equal(a['train_ids'],train); np.testing.assert_array_equal(a['held_ids'],held)
            pred = a['held_prediction'].copy()
        radius,support = data.radius[held-data.nmain],data.support[held-data.nmain]
        assert np.isfinite(pred).all() and not pred[~support].any()
        assert np.all(np.linalg.norm(pred.astype(float),axis=-1) <= radius[:,None]*1.00001+1e-7)
        pieces[(key[2],key[1])].append(dict(ids=held,prediction=pred,cost_scale=np.full(len(held),scale)))
    for item in report['oof_labels']:
        pred,scale = assemble_oof(ids,pieces[(item['arm'],item['seed'])])
        labels = cost_labels(pred,data.target[ids-data.nmain],scale)
        path = ROOT/item['path']; assert file_digest(path) == item['sha256']
        with np.load(path,allow_pickle=False) as a:
            for k,v in dict(ids=ids,prediction=pred,cost_scale=scale,**labels).items(): np.testing.assert_array_equal(a[k],v)
        hashes[item['path']] = item['sha256']
    for path in [*private.joinpath('trials').glob('*.json'),private/'identity.json',
                 *[public/(x+'.json') for x in ('objective_check','input_checks','report','replay','analysis')]]:
        hashes[str(path.relative_to(ROOT))] = file_digest(path)
    child = subprocess.run([sys.executable,'scripts/run_m3w_source_importance_sampling.py',
        '--registration',str(args.registration)],cwd=ROOT,text=True,capture_output=True,check=True)
    event = [json.loads(x) for x in child.stdout.splitlines() if x.startswith('{')][-1]
    assert event['state'] == 'training_complete' and event['new_updates'] == 0
    assert hashes == {path:file_digest(ROOT/path) for path in hashes}
    result = dict(result_source='fresh_run_artifact_checks_cached_verified_controls',
        registration_sha256=file_digest(args.registration),exact_replayed_heads=24,
        weighted_sampling_streams_recomputed_during_replay=24,matched_old_control_draws=24,
        paired_arm_sample_streams=len(pairs),exact_training_factor_vectors=24,
        objective_expectation_real_training_folds=4,oof_archives_recomputed=6,new_training_updates=240000,
        immutable_artifacts=len(hashes),artifact_hashes=hashes,completed_resume=event,new_updates_on_resume=0,
        main_outer_rows_scored=0,new_deployment=False,sensor_asof_certified=False)
    preserve_verification(public/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))


if __name__ == '__main__': main()
