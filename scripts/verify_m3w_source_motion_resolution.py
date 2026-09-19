"""Check replay counts, scores and zero-work completed resume for native motion."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.world_model.m3w_source_motion_resolution import registration, REGISTRATION
from src.evaluation.m3w_experiment_contract import file_digest
from scripts.probe_m3w_source_box_motion import probability_metrics
from scripts.verify_m3w_source_motion_quality import preserve_verification
import numpy as np


def main():
    plan=registration(ROOT); private,public=ROOT/plan['output'],ROOT/plan['reports']
    prep=json.loads((public/'preparation.json').read_text())
    flow=json.loads((public/'extraction.json').read_text())
    probes=json.loads((public/'probes.json').read_text())
    replay=json.loads((public/'probe_replay.json').read_text())
    flow_replay=json.loads((public/'flow_replay.json').read_text())
    assert prep['crops']==prep['exact_reductions']==25300 and prep['rows']==15430
    assert flow_replay['exact_pair_replays']==4*23890
    assert replay['exact_coefficient_replays']==64 and replay['future_poison_queries']==128
    assert replay['illegal_training_roles_rejected']==48
    rawpath=ROOT/plan['raw_label_archive']
    assert file_digest(rawpath)==probes['identity']['raw_labels_sha256']
    with np.load(rawpath,allow_pickle=False) as raw:
        ids=raw['ids'].copy(); target=raw['future_native'].copy()
    exact={'any_nonzero':np.any(target!=0,axis=(1,2)).astype(int),
           'over10_annotation_pixels':(np.sum(target.astype(float)**2,axis=-1).max(1)>100).astype(int)}
    assert exact['over10_annotation_pixels'].sum()==728
    scored=0; labels_checked=0
    for trial in probes['trials']:
        path=ROOT/trial['artifact']['path']; assert file_digest(path)==trial['artifact']['sha256']
        with np.load(path,allow_pickle=False) as a:
            for subset,metric_key in (('train','training'),('held','held')):
                q=a[subset+'_ids']; loc=np.searchsorted(ids,q)
                np.testing.assert_array_equal(ids[loc],q)
                np.testing.assert_array_equal(a[subset+'_label'],exact[trial['label']][loc])
                assert probability_metrics(a[subset+'_label'],a[subset+'_probability'])==trial[metric_key]
                scored+=1; labels_checked+=len(q)
            y,yh=a['train_label'],a['held_label']
            assert probability_metrics(yh,np.full(len(yh),y.mean()))==trial['prior']
            assert not set(a['train_ids']) & set(a['held_ids'])
    paths=[p for p in private.rglob('*') if p.suffix in ('.npy','.npz','.json') and 'heartbeat' not in p.name]
    paths += [public/x for x in ('preparation.json','extraction.json','flow_replay.json','probes.json','probe_replay.json')]
    hashes={str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    resumes=[]
    for script,field in [('prepare_m3w_source_native_motion.py','new_records'),
                         ('build_m3w_source_motion_resolution.py','new_pair_measurements'),
                         ('probe_m3w_source_motion_resolution.py','new_fits')]:
        print(json.dumps(dict(state='completed_resume_check',script=script)),flush=True)
        child=subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,check=True,text=True,capture_output=True)
        event=[json.loads(line) for line in child.stdout.splitlines() if line.startswith('{')][-1]
        assert event['state']=='complete' and event[field]==0
        resumes.append(dict(script=script,event=event))
    assert hashes=={p:file_digest(ROOT/p) for p in hashes}
    result=dict(result_source='fresh_run_verification',registration_sha256=file_digest(ROOT/REGISTRATION),
        exact_crop_reductions=25300,exact_old_control_pairs=23890,exact_flow_replays=4*23890,
        exact_coefficient_replays=64,recomputed_train_held_score_sets=scored,label_checks=labels_checked,
        future_label_poison_queries=128,prohibited_training_role_checks=48,completed_resumes=resumes,
        immutable_artifacts=len(hashes),artifact_hashes=hashes,main_outer_rows_scored=0,
        new_deployment=False,stage5c_executed=False,smc_enabled=False)
    preserve_verification(public/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))


if __name__=='__main__': main()
